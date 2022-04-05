# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
import os
import re
from pathlib import Path

from warp_drive.utils.env_registrar import EnvironmentRegistrar


def get_project_root() -> Path:
    return Path(__file__).absolute().parent.parent.parent


def get_default_env_directory(env_name):
    envs = {
        "TagGridWorld": f"{get_project_root()}"
        f"/example_envs/tag_gridworld/tag_gridworld_step.cu",
        "TagContinuous": f"{get_project_root()}"
        f"/example_envs/tag_continuous/tag_continuous_step.cu",
        "YOUR_ENVIRONMENT": "FULL_PATH_TO_YOUR_ENV_SRC",
    }
    return envs.get(env_name, None)


def update_env_header(
    template_header_file, path=None, num_envs=1, num_agents=1, blocks_per_env=1
):
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
        logging.warning(
            f"the destination header file {destination_header_path}/"
            f"{destination_header_file} already exists; remove and rebuild."
        )
        os.remove(f"{destination_header_path}/{destination_header_file}")

    header_subs = {
        "N_ENVS": str(num_envs),
        "N_AGENTS": str(num_agents),
        "N_BLOCKS_PER_ENV": str(blocks_per_env),
    }
    header_content = ""

    with open(f"{path}/{template_header_file}", "r", encoding="utf8") as reader:
        for line in reader.readlines():
            updated_line = re.sub("<<(.*?)>>", from_dict(header_subs), line)
            header_content += updated_line
    with open(
        f"{destination_header_path}/{destination_header_file}", "w", encoding="utf8"
    ) as writer:
        writer.write(header_content)


def check_env_header(
    header_file="env_config.h", path=None, num_envs=1, num_agents=1, blocks_per_env=1
):

    if path is None:
        path = f"{get_project_root()}/warp_drive/cuda_includes"

    with open(f"{path}/{header_file}", "r", encoding="utf8") as reader:
        for line in reader.readlines():
            if "wkNumberEnvs" in line:
                res = re.findall(r"\b\d+\b", line)
                assert (
                    len(res) == 1 and int(res[0]) == num_envs
                ), f"{header_file} has different num_envs"
            elif "wkNumberAgents" in line:
                res = re.findall(r"\b\d+\b", line)
                assert (
                    len(res) == 1 and int(res[0]) == num_agents
                ), f"{header_file} has different num_agents"
            elif "wkBlocksPerEnv" in line:
                res = re.findall(r"\b\d+\b", line)
                assert (
                    len(res) == 1 and int(res[0]) == blocks_per_env
                ), f"{header_file} has different blocks_per_env"


def update_env_runner(
    template_runner_file,
    path=None,
    env_name=None,
    customized_env_registrar: EnvironmentRegistrar = None,
):
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
        logging.warning(
            f"the destination runner file {destination_runner_path}/"
            f"{destination_runner_file} already exists; remove and rebuild."
        )
        os.remove(f"{destination_runner_path}/{destination_runner_file}")

    env_cuda = None
    if (
        customized_env_registrar is not None
        and customized_env_registrar.get_cuda_env_src_path(env_name) is not None
    ):
        env_cuda = customized_env_registrar.get_cuda_env_src_path(env_name)
        logging.info(
            f"Finding the targeting environment source code "
            f"from the customized environment directory: {env_cuda}"
        )
    elif get_default_env_directory(env_name) is not None:
        env_cuda = get_default_env_directory(env_name)
        logging.info(
            f"Finding the targeting environment source code "
            f"from the default environment directory: {env_cuda}"
        )

    assert env_cuda is not None and isinstance(
        env_cuda, str
    ), "Failed to find or validate the targeting environment"

    runner_subs = {"ENV_CUDA": env_cuda}
    runner_content = ""

    logging.info(
        f"Building the targeting environment "
        f"with source code at: {runner_subs['ENV_CUDA']}"
    )

    with open(f"{path}/{template_runner_file}", "r", encoding="utf8") as reader:
        for line in reader.readlines():
            updated_line = re.sub("<<(.*?)>>", from_dict(runner_subs), line)
            runner_content += updated_line
    with open(
        f"{destination_runner_path}/{destination_runner_file}", "w", encoding="utf8"
    ) as writer:
        writer.write(runner_content)
