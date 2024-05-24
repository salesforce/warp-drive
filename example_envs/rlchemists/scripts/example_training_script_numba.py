# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Example training script for the grid world and continuous versions of Tag.
"""

import argparse
import logging
import os
import sys
import time
import pprint

import torch
import yaml

from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.trainers.trainer_a2c import TrainerA2C

from rlutils.common import get_project_root, load_en_array
from single_agent_one_atom.single_agent_one_atom import CUDASingleAgentOneAtomChemSearch
from single_agent_two_atom.single_agent_two_atom import CUDASingleAgentTwoAtomChemSearch
from warp_drive.utils.env_registrar import EnvironmentRegistrar

_ROOT_DIR = get_project_root()

_SingleAgentOneAtom = [
    "single_agent_one_atom_diffusion2d",
    "single_agent_one_atom_diffusion2d_relaxed",
    "single_agent_one_atom_diffusion3d",
    "single_agent_one_atom_surface",
    "single_agent_one_atom_surface_double_grids",
    "single_agent_one_atom_gas1",
    "single_agent_one_atom_gas2",
    "single_agent_one_atom_gas2_double_grids",
]
_SingleAgentTwoAtom = [
    "single_agent_two_atom_diffusion",
]

# Example usages (from the root folder):
# >> python example_training_script_numba.py --env single_agent_one_atom --type diffusion2d


def setup_trainer_and_train(
    run_configuration,
    en_array=None,
    device_id=0,
    num_devices=1,
    event_messenger=None,
    results_directory=None,
    verbose=True,
):
    """
    Create the environment wrapper, define the policy mapping to agent ids,
    and create the trainer object. Also, perform training.
    """
    logging.getLogger().setLevel(logging.ERROR)

    num_envs = run_configuration["trainer"]["num_envs"]

    # Create a wrapped environment object via the EnvWrapper
    # Ensure that use_cuda is set to True (in order to run on the GPU)
    # ----------------------------------------------------------------
    if run_configuration["name"] in _SingleAgentOneAtom:
        env_registrar = EnvironmentRegistrar()
        env_registrar.add_cuda_env_src_path(CUDASingleAgentOneAtomChemSearch.name,
                                            "single_agent_one_atom.single_agent_one_atom_step_numba",
                                            env_backend="numba")
        env_wrapper = EnvWrapper(
            CUDASingleAgentOneAtomChemSearch(en_array=en_array, **run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            env_registrar=env_registrar,
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] in _SingleAgentTwoAtom:
        env_registrar = EnvironmentRegistrar()
        env_registrar.add_cuda_env_src_path(CUDASingleAgentTwoAtomChemSearch.name,
                                            "single_agent_two_atom.single_agent_two_atom_step_numba",
                                            env_backend="numba")
        env_wrapper = EnvWrapper(
            CUDASingleAgentTwoAtomChemSearch(en_array=en_array, **run_configuration["env"]),
            num_envs=num_envs,
            env_backend="numba",
            env_registrar=env_registrar,
            event_messenger=event_messenger,
            process_id=device_id,
        )
    else:
        raise NotImplementedError(
            f"Currently, the environments supported are ["
            f"{_SingleAgentOneAtom}"
            f"], or"
            f"["
            f"{_SingleAgentTwoAtom}"
            f"]",
        )
    # Policy mapping to agent ids: agents can share models
    # The policy_tag_to_agent_id_map dictionary maps
    # policy model names to agent ids.
    # ----------------------------------------------------
    if len(run_configuration["policy"].keys()) == 1:
        # Using a single (or shared policy) across all agents
        policy_name = list(run_configuration["policy"])[0]
        policy_tag_to_agent_id_map = {
            policy_name: list(env_wrapper.env.agents)
        }
    else:
        # Using different policies for different (sets of) agents
        pass
    # Assert that all the valid policies are mapped to at least one agent
    assert set(run_configuration["policy"].keys()) == set(
        policy_tag_to_agent_id_map.keys()
    )
    # Trainer object
    # --------------
    trainer = TrainerA2C(
        env_wrapper=env_wrapper,
        config=run_configuration,
        policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
        device_id=device_id,
        num_devices=num_devices,
        results_dir=results_directory,
        verbose=verbose,
    )

    # Perform training
    # ----------------
    trainer.train()
    trainer.graceful_close()
    perf_stats = trainer.perf_stats
    print(f"Training steps/s: {perf_stats.steps / perf_stats.total_time:.2f}. \n")


if __name__ == "__main__":

    num_gpus_available = torch.cuda.device_count()
    assert num_gpus_available > 0, "The training script needs a GPU machine to run!"

    # Set logger level e.g., DEBUG, INFO, WARNING, ERROR\n",
    logging.getLogger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        "-e",
        type=str,
        help="the environment to train. This also refers to the"
        "yaml file name in run_configs/.",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        help="the specific environment type to train. This also refers to the"
             "yaml file name in run_configs/.",
    )
    args = parser.parse_args()
    assert args.env is not None, (
        "No env specified. Please use the '-e'- or '--env' option "
        "to specify an environment. The environment name should "
        "match the name of the yaml file in training/run_configs/."
    )

    # Read the run configurations specific to each environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    env_type = args.type
    config_path = os.path.join(
        _ROOT_DIR, "run_configs", f"{args.env}_{env_type}.yaml"
    )
    if not os.path.exists(config_path):
        raise ValueError(
            "Invalid environment specified! The environment name should "
            "match the name of the yaml file in training/run_configs/."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)
        print("Running Configuration: \n")
        pprint.pprint(run_config, sort_dicts=False)

    results_dir = f"{time.time():10.0f}"

    if run_config["name"] in _SingleAgentOneAtom:
        if env_type in ["diffusion2d", "diffusion3d"]:
            en_array = load_en_array(f"{get_project_root()}/en_array/en_array_diffusion.npy")
        elif env_type in ["diffusion2d_relaxed"]:
            en_array = load_en_array(f"{get_project_root()}/en_array/en_array_diffusion_relax.npy")
        elif env_type in ["surface_double_grids", "gas2_double_grids"]:
            en_array = load_en_array(f"{get_project_root()}/en_array/en_array_double_grids.npy")
        else:
            en_array = load_en_array(f"{get_project_root()}/en_array/en_array.npy")
    elif run_config["name"] in _SingleAgentTwoAtom:
        en_array = load_en_array(f"{get_project_root()}/en_array/en_array_2atom_diffusion.npy")
    else:
        en_array = None

    setup_trainer_and_train(run_config, en_array=en_array, results_directory=results_dir)
