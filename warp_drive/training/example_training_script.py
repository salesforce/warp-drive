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

import torch
import yaml

from example_envs.tag_continuous.tag_continuous import TagContinuous
from example_envs.tag_gridworld.tag_gridworld import CUDATagGridWorld
from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.trainer import Trainer
from warp_drive.training.utils.auto_scaler import auto_scaling
from warp_drive.utils.common import get_project_root

_ROOT_DIR = get_project_root()

_TAG_CONTINUOUS = "tag_continuous"
_TAG_GRIDWORLD = "tag_gridworld"

# Example usage (from the root folder):
# >> python warp_drive/training/example_training_script.py --env tag_gridworld
# >> python warp_drive/training/example_training_script.py --env tag_continuous


def setup_trainer_and_train(run_configuration, verbose=True):
    """
    Create the environment wrapper, define the policy mapping to agent ids,
    and create the trainer object. Also, perform training.
    """
    logging.getLogger().setLevel(logging.ERROR)
    torch.cuda.FloatTensor(8)  # add this line for successful cuda_init

    num_envs = run_configuration["trainer"]["num_envs"]

    # Create a wrapped environment object via the EnvWrapper
    # Ensure that use_cuda is set to True (in order to run on the GPU)
    # ----------------------------------------------------------------
    if run_configuration["name"] == _TAG_GRIDWORLD:
        env_wrapper = EnvWrapper(
            CUDATagGridWorld(**run_configuration["env"]),
            num_envs=num_envs,
            use_cuda=True,
        )
    elif run_configuration["name"] == _TAG_CONTINUOUS:
        env_wrapper = EnvWrapper(
            TagContinuous(**run_configuration["env"]), num_envs=num_envs, use_cuda=True
        )
    else:
        raise NotImplementedError(
            f"Currently, the environments supported are ["
            f"{_TAG_GRIDWORLD}, "
            f"{_TAG_CONTINUOUS}"
            f"]",
        )
    # Initialize shared constants for action index to sampled_actions_placeholder
    # ---------------------------------------------------------------------------
    if run_configuration["name"] == _TAG_GRIDWORLD:
        kIndexToActionArr = env_wrapper.env.step_actions
        env_wrapper.env.cuda_data_manager.add_shared_constants(
            {"kIndexToActionArr": kIndexToActionArr}
        )
        env_wrapper.env.cuda_function_manager.initialize_shared_constants(
            env_wrapper.env.cuda_data_manager, constant_names=["kIndexToActionArr"]
        )
    # Policy mapping to agent ids: agents can share models
    # The policy_tag_to_agent_id_map dictionary maps
    # policy model names to agent ids.
    # ----------------------------------------------------
    if len(run_configuration["policy"].keys()) == 1:
        # Using a single (or shared policy) across all agents
        policy_name = list(run_configuration["policy"])[0]
        policy_tag_to_agent_id_map = {
            policy_name: list(env_wrapper.env.taggers) + list(env_wrapper.env.runners)
        }
    else:
        # Using different policies for different(sets) of agents
        policy_tag_to_agent_id_map = {
            "tagger": list(env_wrapper.env.taggers),
            "runner": list(env_wrapper.env.runners),
        }
    # Assert that all the valid policies are mapped to at least one agent
    assert set(run_configuration["policy"].keys()) == set(
        policy_tag_to_agent_id_map.keys()
    )
    # Trainer object
    # --------------
    trainer = Trainer(
        env_wrapper=env_wrapper,
        config=run_configuration,
        policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
        verbose=verbose,
    )

    # Perform training
    # ----------------
    trainer.train()
    trainer.graceful_close()
    perf_stats = trainer.perf_stats
    print(f"Training steps/s: {perf_stats.steps / perf_stats.total_time:.2f}. \n")


if __name__ == "__main__":

    # Set logger level e.g., DEBUG, INFO, WARNING, ERROR\n",
    logging.getLogger().setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str, help="Environment to train.")
    parser.add_argument(
        "--auto_scale",
        "-a",
        action="store_true",
        help="Perform auto (vertical) scaling.",
    )

    args = parser.parse_args()

    # Read the run configurations specific to each environment.
    # Note: The run config yaml(s) can be edited at warp_drive/training/run_configs
    # -----------------------------------------------------------------------------
    assert args.env in [
        _TAG_CONTINUOUS,
        _TAG_GRIDWORLD,
    ], (
        f"Currently, the environment arguments supported "
        f"are ["
        f"{_TAG_GRIDWORLD},"
        f" {_TAG_CONTINUOUS},"
        f"]"
    )

    config_path = os.path.join(
        _ROOT_DIR, "warp_drive", "training", "run_configs", f"{args.env}.yaml"
    )
    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)

    # Perform automatic (vertical) scaling to find the best training parameters.
    if args.auto_scale:
        run_config = auto_scaling(setup_trainer_and_train, run_config)

    setup_trainer_and_train(run_config)
