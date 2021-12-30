# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The Pytorch Lightning Trainer, PerfStats and Metrics classes
"""

import json
import logging
import os
import random
import time

import numpy as np
import torch
import yaml
from gym.spaces import Discrete, MultiDiscrete
from torch import nn

from warp_drive.managers.function_manager import CUDASampler
from warp_drive.training.algorithms.a2c import A2C
from warp_drive.training.algorithms.ppo import PPO
from warp_drive.training.models.fully_connected import FullyConnected
from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.training.utils.param_scheduler import ParamScheduler
from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants

from pytorch_lightning import LightningModule

_ROOT_DIR = get_project_root()

_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_DONE_FLAGS = Constants.DONE_FLAGS
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_COMBINED = "combined"
_EPSILON = 1e-10  # small number to prevent indeterminate divisions


def all_equal(iterable):
    """
    Check all elements of an iterable (e.g., list) are identical
    """
    return len(set(iterable)) <= 1


def recursive_merge_config_dicts(config, default_config):
    """
    Merge the configuration dictionary with the default configuration
    dictionary to fill in any missing configuration keys.
    """
    assert isinstance(config, dict)
    assert isinstance(default_config, dict)

    for k, v in default_config.items():
        if k not in config:
            config[k] = v
        else:
            if isinstance(v, dict):
                recursive_merge_config_dicts(config[k], v)
    return config


class WarpDriveLightningTrainer(LightningModule):

    def __init__(
        self,
        env_wrapper=None,
        config=None,
        policy_tag_to_agent_id_map=None,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
    ):
        """
        Args:
            env_wrapper: the wrapped environment object
            config: the experiment run configuration
            policy_tag_to_agent_id_map:
                a dictionary mapping policy tag to agent ids
            create_separate_placeholders_for_each_policy:
                flag indicating whether there exist separate observations,
                actions and rewards placeholders, as designed in the step
                function. The placeholders will be used in the step() function
                and during training.
                When there's only a single policy, this flag will be False.
                It can also be True when there are multiple policies, yet
                all the agents have the same obs/action space, so we can
                share the same placeholder.
                Defaults to "False"
            obs_dim_corresponding_to_num_agents:
                indicative of which dimension in the observation corresponds
                to the number of agents, as designed in the step function.
                It may be "first" or "last". In other words,
                observations may be shaped (num_agents, *feature_dim) or
                (*feature_dim, num_agents). This is required in order for
                WarpDrive to process the observations correctly.
                Defaults to "first"
        """
        assert env_wrapper is not None
        assert config is not None
        assert isinstance(policy_tag_to_agent_id_map, dict)
        assert len(policy_tag_to_agent_id_map) > 0  # at least one policy
        assert isinstance(create_separate_placeholders_for_each_policy, bool)
        assert obs_dim_corresponding_to_num_agents in ["first", "last"]

        self.cuda_envs = env_wrapper

        # Load in the default configuration
        default_config_path = os.path.join(
            _ROOT_DIR, "warp_drive", "training", "run_configs", "default_configs.yaml"
        )
        with open(default_config_path, "r", encoding="utf8") as fp:
            default_config = yaml.safe_load(fp)

        self.config = config
        # Fill in any missing configuration parameters using the default values
        # Trainer-related configurations
        self.config["trainer"] = recursive_merge_config_dicts(
            self.config["trainer"], default_config["trainer"]
        )
        # Policy-related configurations
        for key in config["policy"]:
            self.config["policy"][key] = recursive_merge_config_dicts(
                self.config["policy"][key], default_config["policy"]
            )
        # Saving-related configurations
        self.config["saving"] = recursive_merge_config_dicts(
            self.config["saving"], default_config["saving"]
        )

        # Note: experiment name reflects run create time.
        experiment_name = f"{time.time():10.0f}"

        # Directory to save model checkpoints and metrics
        self.save_dir = os.path.join(
            self._get_config(["saving", "basedir"]),
            self._get_config(["saving", "name"]),
            self._get_config(["saving", "tag"]),
            experiment_name,
        )
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=False)

        # Policies
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.policies = list(self._get_config(["policy"]).keys())
        self.policies_to_train = [
            policy
            for policy in self.policies
            if self._get_config(["policy", policy, "to_train"])
        ]

        # Flag indicating whether there needs to be separate placeholders / arrays
        # for observation, actions and rewards, for each policy
        self.create_separate_placeholders_for_each_policy = (
            create_separate_placeholders_for_each_policy
        )
        # Note: separate placeholders are needed only when there are
        # multiple policies
        if self.create_separate_placeholders_for_each_policy:
            assert len(self.policies) > 1

        # Number of iterations algebra
        self.num_episodes = self._get_config(["trainer", "num_episodes"])
        self.training_batch_size = self._get_config(["trainer", "train_batch_size"])
        self.num_envs = self._get_config(["trainer", "num_envs"])

        self.training_batch_size_per_env = self.training_batch_size // self.num_envs
        assert self.training_batch_size_per_env > 0

        self.block = self.cuda_envs.cuda_function_manager.block
        self.data_manager = self.cuda_envs.cuda_data_manager
        self.function_manager = self.cuda_envs.cuda_function_manager

        # Seeding
        seed = self.config["trainer"].get("seed", np.int32(time.time()))
        self.cuda_sample_controller = CUDASampler(self.cuda_envs.cuda_function_manager)
        self.cuda_sample_controller.init_random(seed)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Define models, optimizers, and learning rate schedules
        self.models = {}
        self.optimizers = {}
        self.lr_schedules = {}

        # For logging episodic reward
        self.num_completed_episodes = {}
        self.episodic_reward_sum = {}
        self.reward_running_sum = {}

        # Indicates the current timestep of the policy model
        self.current_timestep = {}

        self.total_steps = self.cuda_envs.episode_length * self.num_episodes
        self.num_iters = self.total_steps // self.training_batch_size

        for policy in self.policies:
            self.current_timestep[policy] = 0
            policy_model_config = self._get_config(["policy", policy, "model"])
            if policy_model_config["type"] == "fully_connected":
                self.models[policy] = FullyConnected(
                    self.cuda_envs,
                    policy_model_config["fc_dims"],
                    policy,
                    self.policy_tag_to_agent_id_map,
                    create_separate_placeholders_for_each_policy,
                    obs_dim_corresponding_to_num_agents,
                )
            else:
                raise NotImplementedError

        # Load the model parameters (if model checkpoints are specified)
        # Note: Loading the model checkpoint may also update the current timestep!
        self.load_model_checkpoint()

        for policy in self.policies:
            # Push the models to the GPU
            self.models[policy].cuda()

            # Initialize the (ADAM) optimizer
            policy_config = self._get_config(["policy", policy])
            self.lr_schedules[policy] = ParamScheduler(policy_config["lr"])
            initial_lr = self.lr_schedules[policy].get_param_value(
                timestep=self.current_timestep[policy]
            )
            self.optimizers[policy] = torch.optim.Adam(
                self.models[policy].parameters(), lr=initial_lr
            )

            # Initialize episodic rewards and push to the GPU
            num_agents_for_policy = len(self.policy_tag_to_agent_id_map[policy])
            self.num_completed_episodes[policy] = 0
            self.episodic_reward_sum[policy] = (
                torch.tensor(0).type(torch.float32).cuda()
            )
            self.reward_running_sum[policy] = torch.zeros(
                (self.num_envs, num_agents_for_policy)
            ).cuda()

        # Initialize the trainers
        self.trainers = {}
        self.clip_grad_norm = {}
        self.max_grad_norm = {}
        for policy in self.policies_to_train:
            algorithm = self._get_config(["policy", policy, "algorithm"])
            assert algorithm in ["A2C", "PPO"]
            entropy_coeff = self._get_config(["policy", policy, "entropy_coeff"])
            vf_loss_coeff = self._get_config(["policy", policy, "vf_loss_coeff"])
            self.clip_grad_norm[policy] = self._get_config(
                ["policy", policy, "clip_grad_norm"]
            )
            if self.clip_grad_norm[policy]:
                self.max_grad_norm[policy] = self._get_config(
                    ["policy", policy, "max_grad_norm"]
                )
            normalize_advantage = self._get_config(
                ["policy", policy, "normalize_advantage"]
            )
            normalize_return = self._get_config(["policy", policy, "normalize_return"])
            gamma = policy_config["gamma"]

            if algorithm == "PPO":
                clip_param = self._get_config(["policy", policy, "clip_param"])

            if algorithm == "A2C":
                # Advantage Actor-Critic
                self.trainers[policy] = A2C(
                    discount_factor_gamma=gamma,
                    normalize_advantage=normalize_advantage,
                    normalize_return=normalize_return,
                    vf_loss_coeff=vf_loss_coeff,
                    entropy_coeff=entropy_coeff,
                )
                logging.info(f"Initializing the A2C trainer for policy {policy}")
            elif algorithm == "PPO":
                # Proximal Policy Optimization
                self.trainers[policy] = PPO(
                    discount_factor_gamma=gamma,
                    clip_param=clip_param,
                    normalize_advantage=normalize_advantage,
                    normalize_return=normalize_return,
                    vf_loss_coeff=vf_loss_coeff,
                    entropy_coeff=entropy_coeff,
                )
                logging.info(f"Initializing the PPO trainer for policy {policy}")
            else:
                raise NotImplementedError

        # Push all the data and tensor arrays to the GPU
        # upon resetting environments for the very first time.
        self.cuda_envs.reset_all_envs()

        # Create and push data placeholders to the device
        create_and_push_data_placeholders(
            self.cuda_envs,
            self.policy_tag_to_agent_id_map,
            self.training_batch_size_per_env,
            self.create_separate_placeholders_for_each_policy,
        )

        # Register action placeholders
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                sample_agent_id = self.policy_tag_to_agent_id_map[policy][0]
                action_space = self.cuda_envs.env.action_space[sample_agent_id]

                self._register_actions(action_space, suffix=f"_{policy}")
        else:
            sample_policy = self.policies[0]
            sample_agent_id = self.policy_tag_to_agent_id_map[sample_policy][0]
            action_space = self.cuda_envs.env.action_space[sample_agent_id]

            self._register_actions(action_space)

        # Performance Stats
        self.perf_stats = PerfStats()

        # Metrics
        self.metrics = Metrics()

    def _register_actions(self, action_space, suffix=""):
        if isinstance(action_space, Discrete):
            # Single action
            action_dim = [action_space.n]
        elif isinstance(action_space, MultiDiscrete):
            # Multiple actions
            action_dim = action_space.nvec
        else:
            raise NotImplementedError(
                "Action spaces can be of type" "Discrete or MultiDiscrete"
            )
        if len(action_dim) == 1:
            self.cuda_sample_controller.register_actions(
                self.cuda_envs.cuda_data_manager,
                action_name=_ACTIONS + suffix,
                num_actions=action_dim[0],
            )
        else:
            for action_idx, _ in enumerate(action_dim):
                self.cuda_sample_controller.register_actions(
                    self.cuda_envs.cuda_data_manager,
                    action_name=f"{_ACTIONS}_{action_idx}" + suffix,
                    num_actions=action_dim[action_idx],
                )

    def _evaluate_policies(self, batch_index=0):
        """
        Perform the policy evaluation (forward pass through the models)
        and compute action probabilities
        """
        assert isinstance(batch_index, int)
        probabilities = {}
        for policy in self.policies:
            probabilities[policy], _ = self.models[policy](
                batch_index=batch_index, batch_size=self.training_batch_size_per_env
            )

        # Combine probabilities across policies if there are multiple policies,
        # yet they share the same action placeholders.
        # The action sampler will then need to run just once on each action type.
        if (
            len(self.policies) > 1
            and not self.create_separate_placeholders_for_each_policy
        ):
            # Assert that all the probabilities are of the same length
            # in other words the number of action types for each policy
            # is the same.
            num_action_types = {}
            for policy in self.policies:
                num_action_types[policy] = len(probabilities[policy])
            assert all_equal(list(num_action_types.values()))

            # Initialize combined_probabilities.
            first_policy = list(probabilities.keys())[0]
            num_action_types = num_action_types[first_policy]

            first_action_idx = 0
            num_envs = probabilities[first_policy][first_action_idx].shape[0]
            num_agents = self.cuda_envs.env.num_agents

            combined_probabilities = [None for _ in range(num_action_types)]
            for action_type in range(num_action_types):
                action_dim = probabilities[first_policy][action_type].shape[-1]
                combined_probabilities[action_type] = torch.zeros(
                    (num_envs, num_agents, action_dim)
                ).cuda()

            # Combine the probabilities across policies
            for action_idx in range(num_action_types):
                for policy, prob_values in probabilities.items():
                    agent_to_id_mapping = self.policy_tag_to_agent_id_map[policy]
                    combined_probabilities[action_idx][
                        :, agent_to_id_mapping
                    ] = prob_values[action_idx]

            probabilities = {_COMBINED: combined_probabilities}

        return probabilities

    def forward(self, batch_index, start_event, end_event):

        # Evaluate policies to compute action probabilities
        start_event.record()
        probabilities = self._evaluate_policies(batch_index)
        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.policy_eval_time += (
                start_event.elapsed_time(end_event) / 1000
        )

        # Sample actions using the computed probabilities
        # and push to the batch of actions
        start_event.record()
        self._sample_actions(probabilities, batch_index=batch_index)
        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.action_sample_time += (
                start_event.elapsed_time(end_event) / 1000
        )

        # Step through all the environments
        start_event.record()
        self.cuda_envs.step_all_envs()

        # Bookkeeping rewards and done flags
        _, done_flags = self._bookkeep_rewards_and_done_flags(batch_index)

        # Reset all the environments that are in done state.
        if done_flags.any():
            self.cuda_envs.reset_only_done_envs()

        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.env_step_time += start_event.elapsed_time(end_event) / 1000

    def _generate_rollout_batch(self):
        """
        Generate an environment rollout batch.
        """
        # Code timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for batch_index in range(self.training_batch_size_per_env):
            # Evaluate policies and run step functions
            self.forward(batch_index, start_event, end_event)

    def _generate_training_batch(self):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        metrics_dict = {}

        # Fetch the actions and rewards batches for all agents
        if not self.create_separate_placeholders_for_each_policy:
            all_actions_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_ACTIONS}_batch"
                )
            )
            all_rewards_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_batch"
                )
            )
        done_flags_batch = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            f"{_DONE_FLAGS}_batch"
        )
        # On the device, observations_batch, actions_batch,
        # rewards_batch are all shaped
        # (batch_size, num_envs, num_agents, *feature_dim).
        # done_flags_batch is shaped (batch_size, num_envs)
        # Perform training sequentially for each policy
        for policy in self.policies_to_train:
            if self.create_separate_placeholders_for_each_policy:
                actions_batch = (
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"{_ACTIONS}_batch_{policy}"
                    )
                )
                rewards_batch = (
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"{_REWARDS}_batch_{policy}"
                    )
                )
            else:
                # Filter the actions and rewards only for the agents
                # corresponding to a particular policy
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                actions_batch = all_actions_batch[:, :, agent_ids_for_policy, :]
                rewards_batch = all_rewards_batch[:, :, agent_ids_for_policy]

            # Fetch the (processed) observations batch to pass through the model
            processed_obs_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_PROCESSED_OBSERVATIONS}_batch_{policy}"
                )
            )

            # Policy evaluation for the entire batch
            probabilities_batch, value_functions_batch = self.models[policy](
                obs=processed_obs_batch
            )

            ### TO DO !!!
            ### how to yield the data to the dataloader here ?
            ### we have multi-policies, so the dataset is a dictionary-structure like
            #   {"policy_1": batch, "policy_2": batch}

    def generate_trajectory_samples(self):
        self._generate_rollout_batch()
        self._generate_training_batch()