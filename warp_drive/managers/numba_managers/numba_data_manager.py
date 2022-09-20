# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
from typing import Optional

import numba.cuda as numba_driver
import numpy as np
import torch

from warp_drive.managers.data_manager import CUDADataManager


class NumbaDataManager(CUDADataManager):
    """"""
    """
    Example:
        numba_data_manager = NumbaDataManager(
        num_agents=10, num_envs=5, episode_length=100
        )

        data1 = DataFeed()
        data1.add_data(name="X",
                       data=np.array([[1, 2, 3, 4, 5],
                                      [0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0]])
                      )
        data1.add_data(name="a", data=100)

        numba_data_manager.push_data_to_device(data)

        data2 = DataFeed()

        data2.add_data(name="Y",
                       data=[[0.1,0.2,0.3,0.4,0.5],
                             [0.0,0.0,0.0,0.0,0.0],
                             [0.0,0.0,0.0,0.0,0.0]]
                      )
        numba_data_manager.push_data_to_device(data2, torch_accessible=True)

        X_copy_at_host = numba_data_manager.pull_data_from_device(name="X")

        Y_copy_at_host = numba_data_manager.pull_data_from_device(name="Y")

        if numba_data_manager.is_data_on_device_via_torch("Y"):
            Y_tensor_accessible_by_torch =
            numba_data_manager.data_on_device_via_torch("Y")

        block=(10,1,1)

        grid=(5,1)

        numba_function[grid, block](cuda_data_manager.device_data("X"),
                                    cuda_data_manager.device_data("Y"),)

    """

    def pull_data_from_device(self, name: str):

        assert name in self._host_data
        if name in self._scalar_data_list:
            return self._host_data[name]

        if self.is_data_on_device_via_torch(name):
            return self._device_data_via_torch[name].cpu().numpy()

        else:
            assert name in self._device_data_pointer

            v = self._device_data_pointer[name].copy_to_host()
            return v

    def reset_device(self, name: Optional[str] = None):
        if name is not None:
            assert name in self._device_data_pointer
            assert name in self._host_data

            self._device_data_pointer[name] = numba_driver.to_device(
                self._host_data[name]
            )
        else:
            for name, host_array in self._host_data.items():
                self._device_data_pointer[name] = numba_driver.to_device(host_array)

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
            self._device_data_pointer[name_on_device] = numba_driver.to_device(
                host_array
            )
        else:
            torch_tensor_device = torch.from_numpy(host_array).cuda()
            self._device_data_via_torch[name_on_device] = torch_tensor_device
            self._device_data_pointer[name_on_device] = numba_driver.as_cuda_array(
                torch_tensor_device
            )
