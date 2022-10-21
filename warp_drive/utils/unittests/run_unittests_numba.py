# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
import os
import subprocess

from warp_drive.managers.numba_managers.numba_function_manager import (
    NumbaFunctionManager,
)
from warp_drive.utils.common import get_project_root

content = """
wkNumberEnvs = 2
wkNumberAgents = 5
wkBlocksPerEnv = 1
"""


def create_test_env_config():
    env_config_fname = f"{get_project_root()}/warp_drive/numba_includes/env_config.py"
    if os.path.exists(env_config_fname):
        os.remove(env_config_fname)

    with open(env_config_fname, "w", encoding="utf8") as writer:
        writer.write(content)


if __name__ == "__main__":
    create_test_env_config()
    cmds = [
        f"pytest {get_project_root()}/tests/warp_drive/numba_tests",
        f"pytest {get_project_root()}/tests/example_envs/numba_tests/test_tag_gridworld.py",
        f"pytest {get_project_root()}/tests/example_envs/numba_tests/test_tag_continuous.py",
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
