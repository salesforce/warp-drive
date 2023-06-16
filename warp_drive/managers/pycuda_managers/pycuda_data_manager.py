# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


from typing import Optional

import numpy as np
from warp_drive.utils import autoinit_pycuda
import pycuda.driver as pycuda_driver
import torch

from warp_drive.managers.data_manager import CUDADataManager


class CudaTensorHolder(pycuda_driver.PointerHolderBase):
    """
    A class that facilitates casting tensors to pointers.
    """

    def __init__(self, t):
        super().__init__()
        self.gpudata = t.data_ptr()


class PyCUDADataManager(CUDADataManager):
    """"""
    """
    Example:
        cuda_data_manager = PyCUDADataManager(
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
        super().__init__(
            num_agents=num_agents,
            num_envs=num_envs,
            blocks_per_env=blocks_per_env,
            episode_length=episode_length,
        )

    def pull_data_from_device(self, name: str):

        assert name in self._host_data
        if name in self._scalar_data_list:
            return self._host_data[name]

        if self.is_data_on_device_via_torch(name):
            return self._device_data_via_torch[name].cpu().numpy()

        assert name in self._device_data_pointer

        v = np.empty_like(self._host_data[name])
        pycuda_driver.memcpy_dtoh(v, self._device_data_pointer[name])
        return v

    def reset_device(self, name: Optional[str] = None):
        if name is not None:
            assert name in self._device_data_pointer
            assert name in self._host_data

            device_array_ptr = self._device_data_pointer[name]
            pycuda_driver.memcpy_htod(device_array_ptr, self._host_data[name])
        else:
            for key, host_array in self._host_data.items():
                device_array_ptr = self._device_data_pointer[key]
                pycuda_driver.memcpy_htod(device_array_ptr, host_array)

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
            device_array_ptr = pycuda_driver.mem_alloc(host_array.nbytes)
            pycuda_driver.memcpy_htod(device_array_ptr, host_array)
            self._device_data_pointer[name_on_device] = device_array_ptr
        else:
            torch_tensor_device = torch.from_numpy(host_array).cuda()
            self._device_data_via_torch[name_on_device] = torch_tensor_device
            self._device_data_pointer[name_on_device] = CudaTensorHolder(
                torch_tensor_device
            )
