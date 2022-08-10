# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
import subprocess

from warp_drive.managers.numba_managers.numba_function_manager import NumbaFunctionManager
from warp_drive.utils.common import get_project_root

if __name__ == "__main__":
    cuda_function_manager = NumbaFunctionManager()
    cmds = [
        f"pytest {get_project_root()}/tests/warp_drive/numba_codes",
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
