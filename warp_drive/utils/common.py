# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import os
import re
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent.parent


def get_env_directory(env_name):
    envs = {
        "TagGridWorld": f"{get_project_root()}"
        f"/example_envs/tag_gridworld/tag_gridworld_step.cu",
        "TagContinuous": f"{get_project_root()}"
        f"/example_envs/tag_continuous/tag_continuous_step.cu",
        "YOUR_ENVIRONMENT": "FULL_PATH_TO_YOUR_ENV_SRC",
    }
    return envs.get(env_name)


def update_env_header(template_header_file, path=None, num_envs=1, num_agents=1):
    def from_dict(dct):
        def lookup(match):
            key = match.group(1)
            return dct.get(key, f"<{key} not found>")

        return lookup

    destination_header_path = f"{get_project_root()}/warp_drive/cuda_includes"
    if path is None:
        path = destination_header_path

    destination_header_file = "env_config.h"

    if os.path.exists(f"{destination_header_path}/{destination_header_file}"):
        print(
            f"the destination header file {destination_header_path}/"
            f"{destination_header_file} exist, remove and rebuild "
        )
        os.remove(f"{destination_header_path}/{destination_header_file}")

    header_subs = {"N_ENVS": str(num_envs), "N_AGENTS": str(num_agents)}
    header_content = ""

    with open(f"{path}/{template_header_file}", "r") as reader:
        for line in reader.readlines():
            updated_line = re.sub("<<(.*?)>>", from_dict(header_subs), line)
            header_content += updated_line
    with open(f"{destination_header_path}/{destination_header_file}", "w") as writer:
        writer.write(header_content)


def check_env_header(header_file="env_config.h", path=None, num_envs=1, num_agents=1):

    if path is None:
        path = f"{get_project_root()}/warp_drive/cuda_includes"

    with open(f"{path}/{header_file}", "r") as reader:
        for line in reader.readlines():
            if "num_envs" in line:
                res = re.findall(r"\b\d+\b", line)
                assert (
                    len(res) == 1 and int(res[0]) == num_envs
                ), f"{header_file} has different num_envs"
            elif "num_agents" in line:
                res = re.findall(r"\b\d+\b", line)
                assert (
                    len(res) == 1 and int(res[0]) == num_agents
                ), f"{header_file} has different num_agents"


def update_env_runner(template_runner_file, path=None, env_name=None):
    def from_dict(dct):
        def lookup(match):
            key = match.group(1)
            return dct.get(key, f"<{key} not found>")

        return lookup

    destination_runner_path = f"{get_project_root()}/warp_drive/cuda_includes"
    if path is None:
        path = destination_runner_path

    destination_runner_file = "env_runner.cu"

    if os.path.exists(f"{destination_runner_path}/{destination_runner_file}"):
        print(
            f"the destination runner file {destination_runner_path}/"
            f"{destination_runner_file} exist, remove and rebuild "
        )
        os.remove(f"{destination_runner_path}/{destination_runner_file}")

    runner_subs = {"ENV_CUDA": get_env_directory(env_name)}
    runner_content = ""

    print(
        f"Building the targeting environment "
        f"with source code at: {runner_subs['ENV_CUDA']}"
    )

    with open(f"{path}/{template_runner_file}", "r") as reader:
        for line in reader.readlines():
            updated_line = re.sub("<<(.*?)>>", from_dict(runner_subs), line)
            runner_content += updated_line
    with open(f"{destination_runner_path}/{destination_runner_file}", "w") as writer:
        writer.write(runner_content)
