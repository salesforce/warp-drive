# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
import subprocess

from warp_drive.managers.function_manager import CUDAFunctionManager
from warp_drive.utils.common import get_project_root

if __name__ == "__main__":
    cuda_function_manager = CUDAFunctionManager()
    # the following tests are for blocks_per_env = 1
    # main_file is the source code
    main_file = f"{get_project_root()}/warp_drive/cuda_includes/test_build.cu"
    # cubin_file is the targeted compiled exe
    cubin_file = f"{get_project_root()}/warp_drive/cuda_bin/test_build.fatbin"
    logging.info(f"Compiling {main_file} -> {cubin_file}")
    cuda_function_manager.compile(main_file, cubin_file)

    cmds = [
        f"pytest {get_project_root()}/tests/warp_drive",
        f"pytest {get_project_root()}/tests/example_envs",
    ]
    for cmd in cmds:
        print(f"Running Unit tests: {cmd} ")
        with subprocess.Popen(
            cmd, shell=True, stderr=subprocess.STDOUT
        ) as test_process:
            try:
                outs, errs = test_process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                test_process.kill()
                outs, errs = test_process.communicate()
                logging.error(f"Unit Test Timeout for the test: {cmd}")

    # the following tests are for blocks_per_env = 2
    # main_file is the source code
    main_file = (
        f"{get_project_root()}/warp_drive/cuda_includes/test_build_multiblocks.cu"
    )
    # cubin_file is the targeted compiled exe
    cubin_file = (
        f"{get_project_root()}/warp_drive/cuda_bin/test_build_multiblocks.fatbin"
    )
    logging.info(f"Compiling {main_file} -> {cubin_file}")
    cuda_function_manager.compile(main_file, cubin_file)

    cmds = [
        f"pytest {get_project_root()}/tests/multiblocks_per_env/warp_drive",
        f"pytest {get_project_root()}/tests/multiblocks_per_env/example_envs",
    ]
    for cmd in cmds:
        print(f"Running Unit tests: {cmd} ")
        with subprocess.Popen(
            cmd, shell=True, stderr=subprocess.STDOUT
        ) as test_process:
            try:
                outs, errs = test_process.communicate(timeout=20)
            except subprocess.TimeoutExpired:
                test_process.kill()
                outs, errs = test_process.communicate()
                logging.error(f"Unit Test Timeout for the test: {cmd}")
