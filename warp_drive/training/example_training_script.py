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

import pycuda.driver as cuda_driver
import torch
import yaml

from example_envs.tag_continuous.tag_continuous import TagContinuous
from example_envs.tag_gridworld.tag_gridworld import CUDATagGridWorld
from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.trainer import Trainer
from warp_drive.training.utils.distributed_trainer import perform_distributed_training
from warp_drive.training.utils.vertical_scaler import perform_auto_vertical_scaling
from warp_drive.utils.common import get_project_root

_ROOT_DIR = get_project_root()

_TAG_CONTINUOUS = "tag_continuous"
_TAG_GRIDWORLD = "tag_gridworld"

# Example usages (from the root folder):
# >> python warp_drive/training/example_training_script.py -e tag_gridworld
# >> python warp_drive/training/example_training_script.py --env tag_continuous


def setup_trainer_and_train(
    run_configuration,
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
            event_messenger=event_messenger,
            process_id=device_id,
        )
    elif run_configuration["name"] == _TAG_CONTINUOUS:
        env_wrapper = EnvWrapper(
            TagContinuous(**run_configuration["env"]),
            num_envs=num_envs,
            use_cuda=True,
            event_messenger=event_messenger,
            process_id=device_id,
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
        # Using different policies for different (sets of) agents
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

    num_gpus_available = cuda_driver.Device.count()
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
        "--auto_scale",
        "-a",
        action="store_true",
        help="perform auto scaling.",
    )
    parser.add_argument(
        "--num_gpus",
        "-n",
        type=int,
        default=-1,
        help="the number of GPU devices for (horizontal) scaling, "
        "default=-1 (using configure setting)",
    )
    parser.add_argument(
        "--results_dir", type=str, help="name of the directory to save results into."
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
    config_path = os.path.join(
        _ROOT_DIR, "warp_drive", "training", "run_configs", f"{args.env}.yaml"
    )
    if not os.path.exists(config_path):
        raise ValueError(
            "Invalid environment specified! The environment name should "
            "match the name of the yaml file in training/run_configs/."
        )

    with open(config_path, "r", encoding="utf8") as fp:
        run_config = yaml.safe_load(fp)

    if args.auto_scale:
        # Automatic scaling
        print("Performing Auto Scaling!\n")
        # First, perform vertical scaling.
        run_config = perform_auto_vertical_scaling(setup_trainer_and_train, run_config)
        # Next, perform horizontal scaling.
        # Set `num_gpus` to the maximum number of GPUs available
        run_config["trainer"]["num_gpus"] = num_gpus_available
        print(f"We will be using {num_gpus_available} GPU(s) for training.")
    elif args.num_gpus >= 1:
        # Set the appropriate num_gpus configuration parameter
        if args.num_gpus <= num_gpus_available:
            print(f"We have successfully found {args.num_gpus} GPUs!")
            run_config["trainer"]["num_gpus"] = args.num_gpus
        else:
            print(
                f"You requested for {args.num_gpus} GPUs, but we were only able to "
                f"find {num_gpus_available} GPU(s)! \nDo you wish to continue? [Y/n]"
            )
            if input() != "Y":
                print("Terminating program.")
                sys.exit()
            else:
                run_config["trainer"]["num_gpus"] = num_gpus_available
    elif "num_gpus" not in run_config["trainer"]:
        run_config["trainer"]["num_gpus"] = 1

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        results_dir = f"{time.time():10.0f}"

    print(f"Training with {run_config['trainer']['num_gpus']} GPU(s).")
    if run_config["trainer"]["num_gpus"] > 1:
        perform_distributed_training(setup_trainer_and_train, run_config, results_dir)
    else:
        setup_trainer_and_train(run_config, results_directory=results_dir)
