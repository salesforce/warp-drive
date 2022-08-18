# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from typing import Optional

import logging
import numpy as np
import torch

import numba.cuda as numba_driver

from warp_drive.managers.data_manager import CUDADataManager


class NumbaDataManager(CUDADataManager):

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

            self._device_data_pointer[name] = numba_driver.to_device(self._host_data[name])
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
            self._device_data_pointer[name_on_device] = numba_driver.to_device(host_array)
        else:
            torch_tensor_device = torch.from_numpy(host_array).cuda()
            self._device_data_via_torch[name_on_device] = torch_tensor_device
            self._device_data_pointer[name_on_device] = numba_driver.as_cuda_array(torch_tensor_device)