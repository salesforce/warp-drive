# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging

import pycuda.autoinit
from pycuda.driver import Context


class DeviceArchitectures:
    """
    Reference:
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities"
    """

    MaxBlocksPerSM = {
        "sm_35": 16,
        "sm_37": 16,
        "sm_50": 32,
        "sm_52": 32,
        "sm_53": 32,
        "sm_60": 32,
        "sm_61": 32,
        "sm_62": 32,
        "sm_70": 32,
        "sm_72": 32,
        "sm_75": 16,
        "sm_80": 32,
        "sm_86": 16,
        "sm_87": 16,
    }
    MaxThreadsPerSM = {
        "sm_35": 2048,
        "sm_37": 2048,
        "sm_50": 2048,
        "sm_52": 2048,
        "sm_53": 2048,
        "sm_60": 2048,
        "sm_61": 2048,
        "sm_62": 2048,
        "sm_70": 2048,
        "sm_72": 2048,
        "sm_75": 1024,
        "sm_80": 2048,
        "sm_86": 1536,
        "sm_87": 1536,
    }


def calculate_blocks_per_env(num_agents):
    max_threads_per_block = Context.get_device().max_threads_per_block

    return (num_agents - 1) // max_threads_per_block + 1


def validate_device_setup(arch, num_blocks, threads_per_block, blocks_per_env):
    try:
        # max_blocks_per_multiprocessor is only supported after CUDA 11.0 build
        max_blocks_per_sm = Context.get_device().max_blocks_per_multiprocessor
    except AttributeError:
        max_blocks_per_sm = DeviceArchitectures.MaxBlocksPerSM.get(arch, None)
    try:
        max_threads_per_sm = Context.get_device().max_threads_per_multiprocessor
    except AttributeError:
        max_threads_per_sm = DeviceArchitectures.MaxThreadsPerSM.get(arch, None)
    try:
        num_sm = Context.get_device().multiprocessor_count
    except Exception as err:
        logging.error(err)
        num_sm = None

    if max_blocks_per_sm is None or max_threads_per_sm is None or num_sm is None:
        raise Exception("Unknown GPU architecture.")
    max_blocks_by_threads = int((max_threads_per_sm - 1) // threads_per_block + 1)
    available_blocks_per_sm = min(max_blocks_per_sm, max_blocks_by_threads)
    max_blocks = available_blocks_per_sm * num_sm

    if max_blocks < num_blocks:
        if blocks_per_env == 1:
            logging.warning(
                f"Warning: max number of blocks available for simultaneous "
                f"run is {max_blocks}, "
                f"however, the number of blocks requested is {num_blocks}. "
                f"Therefore, the simulation will likely under-perform"
            )
        else:
            logging.warning(
                f"Warning: max number of blocks available for simultaneous "
                f"run is {max_blocks}, "
                f"however, the number of blocks requested is {num_blocks}. "
                f"Since blocks_per_env > 1, block synchronization scheduling "
                f"can cause a dead-lock "
            )
            return False

    return True
