# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
from typing import Dict, Optional

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda_driver
import torch

from warp_drive.utils.data_feed import DataFeed


class CudaTensorHolder(pycuda.driver.PointerHolderBase):
    """
    A class that facilitates casting tensors to pointers.
    """

    def __init__(self, t):
        super().__init__()
        self.gpudata = t.data_ptr()


class CUDADataManager:
    """
    CUDA Data Manager: manages the data initialization of GPU,
    and data transfer between CPU host and GPU device

    Example:
        cuda_data_manager = CUDADataManager(
        num_agents=10, num_envs=5, episode_length=100
        )

        data1 = DataFeed()
        data1.add_data(name="X", data=np.array([[1, 2, 3, 4, 5],
                                               [0, 0, 0, 0, 0],
                                               [0, 0, 0, 0, 0]])
                      )
        data1.add_data(name="a", data=100)
        cuda_data_manager.push_data_to_device(data)

        data2 = DataFeed()
        data2.add_data(name="Y", data=[[0.1,0.2,0.3,0.4,0.5],
                                      [0.0,0.0,0.0,0.0,0.0],
                                      [0.0,0.0,0.0,0.0,0.0]]
                      )
        cuda_data_manager.push_data_to_device(data2, torch_accessible=True)

        X_copy_at_host = cuda_data_manager.pull_data_from_device(name="X")
        Y_copy_at_host = cuda_data_manager.pull_data_from_device(name="Y")

        if cuda_data_manager.is_data_on_device_via_torch("Y"):
            Y_tensor_accessible_by_torch =
            cuda_data_manager.data_on_device_via_torch("Y")

        # cuda_function here assumes a compiled CUDA C function
        cuda_function(cuda_data_manager.device_data("X"),
                      cuda_data_manager.device_data("Y"),
                      block=(10,1,1), grid=(5,1))


    """

    def __init__(
        self,
        num_agents: int = None,
        num_envs: int = None,
        blocks_per_env: int = 1,
        episode_length: int = None,
    ):
        """
        :param num_agents: total number of agents for each env
        :param num_envs: total number of example_envs running in parallel
        :param blocks_per_env: number of blocks to cover one environment
        :param episode_length: the length of one single episode, used for logging in
        general, but in many example_envs, it also sets the duration of a game.

        self._meta_info: maintains meta information as the scalar integers or floats
        self._device_data_pointer: maintains CUDA pointers
        to the data array in the device
        self._device_data_via_torch: maintains the torch tensor
        in the device accessed directly by torch
        self._shared_constant: maintains the shared constants
        that are going to be initialized in the CUDA runtime
        """

        self._meta_info = {}
        self._host_data = {}
        self._device_data_pointer = {}
        self._scalar_data_list = []
        self._reset_data_list = []
        self._log_data_list = []
        self._device_data_via_torch = {}
        self._shared_constants = {}

        self._shape = {}
        self._dtype = {}

        assert num_agents is not None
        assert num_envs is not None
        assert blocks_per_env is not None
        assert episode_length is not None

        self.add_meta_info(
            {
                "n_agents": num_agents,
                "episode_length": episode_length,
                "n_envs": num_envs,
                "blocks_per_env": blocks_per_env,
            }
        )
        self._add_log_mask_and_push_to_device()
        self._add_done_and_push_to_device()

    def _add_log_mask_and_push_to_device(self):
        """
        log_mask of length: episode_length + 1
        log_mask = [0, 0, 0, 0, 0 ....], where the 0th entry
        is reserved for the starting position

        """
        assert "episode_length" in self._meta_info, (
            "self.meta_info['episode_length'] "
            "is required to expand the time dimension as requested by "
            "the dense log mask "
        )
        log_mask = np.zeros(self._meta_info["episode_length"] + 1, dtype=np.int32)
        data = DataFeed()
        data.add_data(name="_log_mask_", data=log_mask)
        self.push_data_to_device(data)

    def _add_done_and_push_to_device(self):
        """
        done flag for each env
        """

        done = np.zeros(self._meta_info["n_envs"], dtype=np.int32)
        done_data = DataFeed()
        done_data.add_data(name="_done_", data=done)
        self.push_data_to_device(done_data, torch_accessible=True)

        timestep = np.zeros(self._meta_info["n_envs"], dtype=np.int32)
        timestep_data = DataFeed()
        timestep_data.add_data(name="_timestep_", data=timestep)
        self.push_data_to_device(timestep_data, torch_accessible=False)

    def add_meta_info(self, meta: Dict):
        """
        Add meta information to the data manager, only accepts scalar integer or float

        :param meta: for example, {"episode_length": 100, "num_agents": 10}
        """
        assert isinstance(meta, dict)

        for key, value in meta.items():
            assert (
                key not in self._meta_info
            ), f"the meta info with name: {key} has already been registered"
            assert not isinstance(
                list, np.ndarray
            ), "the meta info only accepts scalar value not list or array"
            assert isinstance(
                value, (int, np.integer, float, np.floating)
            ), "the meta needs to be casted to a float or an int"
            if isinstance(value, (int, np.integer)):
                self._meta_info[key] = np.int32(value)
            elif isinstance(value, (float, np.floating)):
                self._meta_info[key] = np.float32(value)

    def add_shared_constants(self, constants: Dict):
        """
        Add shared constants to the data manager

        :param constants: e.g., {"action_mapping": [[0,0], [1,1], [-1,-1]]}
        """

        for key, value in constants.items():
            assert key not in self._shared_constants, (
                f"the data with name: {key} has "
                f"already been added as the shared constant"
            )
            if isinstance(value, (np.ndarray, list)):
                if isinstance(value, np.ndarray):
                    if not value.flags.c_contiguous:
                        array = np.array(value, order="C")
                        assert array.flags.c_contiguous
                        self._type_warning_helper(
                            key,
                            old="F_CONTIGUOUS",
                            new="C_CONTIGUOUS",
                            comment="C_CONTIGUOUS(row major index "
                            "is required for CUDA C/C++)",
                        )
                    else:
                        array = value
                elif isinstance(value, list):
                    array = np.array(value, order="C")
                else:
                    raise ValueError(
                        f"the data '{key}' needs to be cast to a list or an array"
                    )

                # only accepts 32bits
                if array.dtype.name == "float64":
                    array = array.astype("float32")
                    self._type_warning_helper(key, old="float64", new="float32")
                elif array.dtype.name == "int64":
                    array = array.astype("int32")
                    self._type_warning_helper(key, old="int64", new="int32")

                self._shared_constants[key] = array

                self._shape[key] = self._shared_constants[key].shape
                self._dtype[key] = self._shared_constants[key].dtype.name
                self._shape_info_helper(
                    key, dtype=array.dtype.name, shape=self._shared_constants[key].shape
                )

            elif isinstance(value, (int, np.integer, float, np.floating)):
                if isinstance(value, (int, np.integer)):
                    self._shared_constants[key] = np.int32(value)
                elif isinstance(value, (float, np.floating)):
                    self._shared_constants[key] = np.float32(value)
                self._shape[key] = ()
                self._dtype[key] = self._shared_constants[key].dtype.name

            else:
                raise ValueError(
                    f"the shared constant '{key}' needs to be cast to a "
                    f"float, int, list or array"
                )

    def push_data_to_device(self, data: Dict, torch_accessible: bool = False):
        """
        Register data to the host, and push to the device
        (1) register at self._host_data
        (2) push to device and register at self._device_data_pointer,
        CUDA program can directly access those data via pointer
        (3) if save_copy_and_apply_at_reset or log_data_across_episode
        as instructed by the data, register and push to device using step (1)(2) too

        :param data: e.g., {"name": {"data": numpy array,
        "attributes": {"save_copy_and_apply_at_reset": True,
        "log_data_across_episode": True}}}.
        This data dictionary can be constructed by warp_drive.utils.data_feed.DataFeed
        :param torch_accessible: if True, the data is directly accessible by Pytorch
        """
        assert isinstance(data, dict)

        logging.info("\nPushing data to device...")
        for key, content in data.items():
            assert (
                key not in self._host_data
            ), f"the data with name: {key} has already been registered at the host"
            value = content["data"]
            save_copy_and_apply_at_reset = content["attributes"][
                "save_copy_and_apply_at_reset"
            ]
            log_data_across_episode = content["attributes"]["log_data_across_episode"]

            if isinstance(value, (np.ndarray, list)):

                assert key not in self._device_data_pointer, (
                    f"the data with name: {key} has " f"already been pushed to device"
                )

                if isinstance(value, np.ndarray):
                    if not value.flags.c_contiguous:
                        array = np.array(value, order="C")
                        assert array.flags.c_contiguous
                        self._type_warning_helper(
                            key,
                            old="F_CONTIGUOUS",
                            new="C_CONTIGUOUS",
                            comment="C_CONTIGUOUS(row major index "
                            "is required for CUDA C/C++)",
                        )
                    else:
                        array = value
                elif isinstance(value, list):
                    array = np.array(value, order="C")
                else:
                    raise ValueError(
                        f"the data '{key}' needs to be cast to a list or an array"
                    )

                # only accepts 32bits
                if array.dtype.name == "float64":
                    array = array.astype("float32")
                    self._type_warning_helper(key, old="float64", new="float32")
                elif array.dtype.name == "int64":
                    array = array.astype("int32")
                    self._type_warning_helper(key, old="int64", new="int32")

                self._host_data[key] = array
                self._to_device(
                    name=key, name_on_device=None, torch_accessible=torch_accessible
                )

                self._shape[key] = self._host_data[key].shape
                self._dtype[key] = self._host_data[key].dtype.name
                self._shape_info_helper(
                    key, dtype=array.dtype.name, shape=self._host_data[key].shape
                )

                if save_copy_and_apply_at_reset:
                    assert key not in self._reset_data_list, (
                        f"the data with name: {key} has "
                        f"already been registered at the reset_data_list"
                    )
                    key_at_reset = f"{key}_at_reset"
                    self._shape[key_at_reset] = self._host_data[key].shape
                    self._dtype[key_at_reset] = self._host_data[key].dtype.name
                    self._shape_info_helper(
                        key_at_reset,
                        dtype=array.dtype.name,
                        shape=self._host_data[key].shape,
                    )
                    self._to_device(
                        key,
                        name_on_device=key_at_reset,
                        torch_accessible=torch_accessible,
                    )
                    self._reset_data_list.append(key)

                if log_data_across_episode:
                    assert not torch_accessible, (
                        "log_data_across_episode=True is not "
                        "supported for the data that have torch_accessible=True"
                    )

                    assert key not in self._log_data_list, (
                        f"the data with name: {key} has "
                        f"already been registered at the log_data_list"
                    )
                    assert "episode_length" in self._meta_info, (
                        "self.meta_info['episode_length'] "
                        "is required to expand the time dimension to "
                        "create a log buffer"
                    )
                    assert array.shape[0] == self.meta_info("n_envs")
                    assert array.shape[1] == self.meta_info("n_agents")
                    array_for_log = np.zeros(
                        shape=(self.meta_info("episode_length") + 1, *array[0].shape),
                        dtype=array.dtype,
                    )
                    key_for_log = f"{key}_for_log"
                    self._host_data[key_for_log] = array_for_log
                    self._shape[key_for_log] = self._host_data[key_for_log].shape
                    self._dtype[key_for_log] = self._host_data[key_for_log].dtype.name
                    self._shape_info_helper(
                        key_for_log,
                        dtype=array_for_log.dtype.name,
                        shape=self._host_data[key_for_log].shape,
                    )

                    self._to_device(key_for_log, name_on_device=key_for_log)
                    self._log_data_list.append(key)

            # for scalar int or float, no need to assign CUDA memory to it
            elif isinstance(value, (int, np.integer, float, np.floating)):

                assert key not in self._scalar_data_list, (
                    f"the data with name: {key} has " f"already been pushed to device"
                )

                if isinstance(value, (int, np.integer)):
                    self._host_data[key] = np.int32(value)
                elif isinstance(value, (float, np.floating)):
                    self._host_data[key] = np.float32(value)
                self._shape[key] = ()
                self._dtype[key] = self._host_data[key].dtype.name
                self._shape_info_helper(
                    key,
                    dtype=self._dtype[key],
                    shape=self._shape[key],
                )
                self._scalar_data_list.append(key)
            else:
                raise ValueError(
                    f"the data '{key}' needs to be casted to a"
                    f" float, int, list or array"
                )

    def pull_data_from_device(self, name: str):
        """
        Fetch the values of device array back to the host

        :param name: name of the device array
        returns: a host copy of scalar data or numpy array
        fetched back from the device array
        """

        assert name in self._host_data
        if name in self._scalar_data_list:
            return self._host_data[name]

        if self.is_data_on_device_via_torch(name):
            return self._device_data_via_torch[name].cpu().numpy()

        assert name in self._device_data_pointer

        v = np.empty_like(self._host_data[name])
        cuda_driver.memcpy_dtoh(v, self._device_data_pointer[name])
        return v

    def data_on_device_via_torch(self, name: str) -> torch.Tensor:
        """
        The data on the device. This is used for Pytorch default access within GPU.
        To fetch the tensor back to the host,
        call pull_data_from_device()

        :param name: name of the device array
        returns: the tensor itself at the device.
        """
        assert name in self._device_data_via_torch

        return self._device_data_via_torch[name]

    def reset_device(self, name: Optional[str] = None):
        """
        Reset the device array values back to the host array values
        Note: this reset is not a device-only execution,
        but incurs data transfer from host to device

        :param name: (optional) reset a device array by name, if None, reset all arrays
        """
        if name is not None:
            assert name in self._device_data_pointer
            assert name in self._host_data

            device_array_ptr = self._device_data_pointer[name]
            cuda_driver.memcpy_htod(device_array_ptr, self._host_data[name])
        else:
            for key, host_array in self._host_data.items():
                device_array_ptr = self._device_data_pointer[key]
                cuda_driver.memcpy_htod(device_array_ptr, host_array)

    def meta_info(self, name: str):
        assert name in self._meta_info

        return self._meta_info[name]

    def shared_constant(self, name: str):
        assert name in self._shared_constants

        return self._shared_constants[name]

    def device_data(self, name: str):
        """
        :param name: name of the device data
        returns: the data pointer in the device for CUDA to access
        """
        if name in self._scalar_data_list:
            assert name in self._host_data
            return self._host_data[name]
        assert name in self._device_data_pointer
        return self._device_data_pointer[name]

    def _to_device(
        self,
        name: str,
        name_on_device: Optional[str] = None,
        torch_accessible: bool = False,
    ):
        assert name in self._host_data
        host_array = self._host_data[name]
        if name_on_device is None:
            name_on_device = name
        assert name_on_device not in self._device_data_pointer
        if not torch_accessible:
            device_array_ptr = cuda_driver.mem_alloc(host_array.nbytes)
            cuda_driver.memcpy_htod(device_array_ptr, host_array)
            self._device_data_pointer[name_on_device] = device_array_ptr
        else:
            torch_tensor_device = torch.from_numpy(host_array).cuda()
            self._device_data_via_torch[name_on_device] = torch_tensor_device
            self._device_data_pointer[name_on_device] = CudaTensorHolder(
                torch_tensor_device
            )

    def is_data_on_device(self, name: str) -> bool:
        return name in self._device_data_pointer

    def is_data_on_device_via_torch(self, name: str) -> bool:
        """
        This is used to check if the data exist and accessible
        via Pytorch default access within GPU.
        name: name of the device
        """
        return self.is_data_on_device(name) and (name in self._device_data_via_torch)

    def get_shape(self, name: str):
        assert name in self._shape
        return self._shape[name]

    def get_dtype(self, name: str):
        assert name in self._dtype
        return self._dtype[name]

    @staticmethod
    def _type_warning_helper(key: str, old: str, new: str, comment=None):
        logging.warning(
            f"CUDADataManager casts the data '{key}' " f"from type {old} to {new}"
        )
        if comment:
            logging.warning(comment)

    @staticmethod
    def _shape_info_helper(key: str, dtype: str, shape):
        logging.info(f"- {key:<80}: dtype={dtype:<10}, shape={shape}")

    @property
    def host_data(self):
        return self._host_data

    @property
    def scalar_data_list(self):
        return self._scalar_data_list

    @property
    def reset_data_list(self):
        return self._reset_data_list

    @property
    def log_data_list(self):
        return self._log_data_list
