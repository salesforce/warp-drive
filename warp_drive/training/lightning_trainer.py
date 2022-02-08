# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The PytorchLightning-based Trainer, PerfStats and Metrics classes
"""

import argparse
import json
import logging
import os
import random
import time
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import yaml
from gym.spaces import Discrete, MultiDiscrete
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from warp_drive.managers.function_manager import CUDASampler
from warp_drive.training.algorithms.a2c import A2C
from warp_drive.training.algorithms.ppo import PPO
from warp_drive.training.models.fully_connected import FullyConnected
from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.training.utils.param_scheduler import ParamScheduler
from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants

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


class WarpDriveTrainer(LightningModule):
    """
    The trainer object using PytorchLightning training APIs.
    """

    def __init__(
        self,
        env_wrapper=None,
        config=None,
        policy_tag_to_agent_id_map=None,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
        **kwargs,
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
        super().__init__()
        # Disable automatic optimization
        self.automatic_optimization = False
        self.avg_reward = {}

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
        self.policies = sorted(list(self._get_config(["policy"]).keys()))
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

        # Define models
        self.models = {}

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
            policy_config = self._get_config(["policy", policy])
            algorithm = policy_config["algorithm"]
            assert algorithm in ["A2C", "PPO"]
            entropy_coeff = policy_config["entropy_coeff"]
            vf_loss_coeff = policy_config["vf_loss_coeff"]
            self.clip_grad_norm[policy] = policy_config["clip_grad_norm"]
            if self.clip_grad_norm[policy]:
                self.max_grad_norm[policy] = policy_config["max_grad_norm"]
            normalize_advantage = policy_config["normalize_advantage"]
            normalize_return = policy_config["normalize_return"]
            gamma = policy_config["gamma"]

            if algorithm == "PPO":
                clip_param = policy_config["clip_param"]

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

    def _get_config(self, args):
        assert isinstance(args, (tuple, list))
        config = self.config
        for arg in args:
            try:
                config = config[arg]
            except ValueError:
                logging.error("Missing configuration '{arg}'!")
        return config

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

    def _sample_actions(self, probabilities, batch_index=0):
        """
        Sample action probabilities (and push the sampled actions to the device).
        """
        assert isinstance(batch_index, int)
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                suffix = f"_{policy}"
                self._sample_actions_helper(
                    probabilities[policy], batch_index, suffix=suffix
                )
        else:
            assert len(probabilities) == 1
            policy = list(probabilities.keys())[0]
            self._sample_actions_helper(probabilities[policy], batch_index)

    def _sample_actions_helper(self, probabilities, batch_index, suffix=""):

        num_action_types = len(probabilities)

        if num_action_types == 1:
            action_name = _ACTIONS + suffix
            self.cuda_sample_controller.sample(
                self.cuda_envs.cuda_data_manager, probabilities[0], action_name
            )
            # Push actions to the batch of actions
            actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                action_name
            )
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                name=f"{_ACTIONS}_batch" + suffix
            )[batch_index] = actions

        else:
            for action_idx, probs in enumerate(probabilities):
                action_name = f"{_ACTIONS}_{action_idx}" + suffix
                self.cuda_sample_controller.sample(
                    self.cuda_envs.cuda_data_manager, probs, action_name
                )
                # Push (indexed) actions to 'actions' and 'actions_batch'
                actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    action_name
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=_ACTIONS + suffix
                )[:, :, action_idx] = actions
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_ACTIONS}_batch" + suffix
                )[batch_index, :, :, action_idx] = actions

    def _bookkeep_rewards_and_done_flags(self, batch_index):
        """
        Push rewards and done flags to the corresponding batched versions.
        Also, update the episodic reward
        """
        assert isinstance(batch_index, int)
        # Push done flags to done_flags_batch
        done_flags = (
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch("_done_") > 0
        )
        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            name=f"{_DONE_FLAGS}_batch"
        )[batch_index] = done_flags

        done_env_ids = done_flags.nonzero()

        # Push rewards to rewards_batch and update the episodic rewards
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_{policy}"
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_REWARDS}_batch_{policy}"
                )[batch_index] = rewards

                # Update the episodic rewards
                self._update_episodic_rewards(rewards, done_env_ids, policy)

        else:
            rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                _REWARDS
            )
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                name=f"{_REWARDS}_batch"
            )[batch_index] = rewards

            # Update the episodic rewards
            # (sum of individual step rewards over an episode)
            for policy in self.policies:
                self._update_episodic_rewards(
                    rewards[:, self.policy_tag_to_agent_id_map[policy]],
                    done_env_ids,
                    policy,
                )
        return rewards, done_flags

    def _update_episodic_rewards(self, rewards, done_env_ids, policy):
        self.reward_running_sum[policy] += rewards

        num_completed_episodes = len(done_env_ids)
        if num_completed_episodes > 0:
            # Update the episodic rewards
            self.episodic_reward_sum[policy] += torch.sum(
                self.reward_running_sum[policy][done_env_ids]
            )
            self.num_completed_episodes[policy] += num_completed_episodes
            # Reset the reward running sum
            self.reward_running_sum[policy][done_env_ids] = 0

    def _log_metrics(self, metrics):
        # Log the metrics if it is not empty
        if len(metrics) > 0:
            perf_stats = self.perf_stats.get_perf_stats()

            print("\n")
            print("=" * 40)
            print(
                f"{'Iterations Completed':40}: "
                f"{self.perf_stats.iters} / {self.num_iters}"
            )
            self.perf_stats.pretty_print(perf_stats)
            self.metrics.pretty_print(metrics)
            print("=" * 40, "\n")

            # Log metrics and performance stats
            logs = {"Iterations Completed": self.perf_stats.iters}
            logs.update(metrics)
            logs.update({"Perf. Stats": perf_stats})

            results_filename = os.path.join(self.save_dir, "results.json")
            print(f"Saving the results to the file '{results_filename}'")
            with open(results_filename, "a+", encoding="utf8") as fp:
                json.dump(logs, fp)
                fp.write("\n")

    def load_model_checkpoint(self, ckpts_dict=None):
        """
        Load the model parameters if a checkpoint path is specified.
        """
        if ckpts_dict is None:
            logging.info(
                "Loading trainer model checkpoints from the run configuration."
            )
            for policy in self.policies:
                ckpt_filepath = self.config["policy"][policy]["model"][
                    "model_ckpt_filepath"
                ]
                self._load_model_checkpoint_helper(policy, ckpt_filepath)
        else:
            assert isinstance(ckpts_dict, dict)
            print("Loading the provied trainer model checkpoints.")
            for policy, ckpt_filepath in ckpts_dict.items():
                assert policy in self.policies
                self._load_model_checkpoint_helper(policy, ckpt_filepath)

    def _load_model_checkpoint_helper(self, policy, ckpt_filepath):
        if ckpt_filepath != "":
            assert os.path.isfile(ckpt_filepath), "Invalid model checkpoint path!"
            print(
                f"Loading the '{policy}' torch model "
                f"from the previously saved checkpoint: '{ckpt_filepath}'"
            )
            self.models[policy].load_state_dict(torch.load(ckpt_filepath))

            # Update the current timestep using the saved checkpoint filename
            timestep = int(ckpt_filepath.split(".state_dict")[0].split("_")[-1])
            print(f"Updating the timestep for the '{policy}' model to {timestep}.")
            self.current_timestep[policy] = timestep

    def save_model_checkpoint(self, iteration=0):
        """
        Save the model parameters
        """
        # Save model checkpoints if specified (and also for the last iteration)
        if (
            iteration % self.config["saving"]["model_params_save_freq"] == 0
            or iteration == self.num_iters - 1
        ):
            for policy, model in self.models.items():
                filepath = os.path.join(
                    self.save_dir,
                    f"{policy}_{self.current_timestep[policy]}.state_dict",
                )
                print(f"Saving the '{policy}' torch model to the file: '{filepath}'.")

                torch.save(model.state_dict(), filepath)

    def graceful_close(self):
        # Delete the sample controller to clear
        # the random seeds defined in the CUDA memory heap
        del self.cuda_sample_controller
        print("Trainer exits gracefully")

    def fetch_episode_states(
        self,
        list_of_states=None,  # list of states (data array names) to fetch
        env_id=0,  # environment id to fetch the states from
    ):
        """
        Step through env and fetch the desired states (data arrays on the GPU)
        for an entire episode. The trained models will be used for evaluation.
        """
        assert 0 <= env_id < self.num_envs
        assert list_of_states is not None
        assert isinstance(list_of_states, list)
        assert len(list_of_states) > 0

        logging.info(f"Fetching the episode states: {list_of_states} from the GPU.")
        # Ensure env is reset before the start of training, and done flags are False
        self.cuda_envs.reset_all_envs()
        env = self.cuda_envs.env

        episode_states = {}

        for state in list_of_states:
            assert self.cuda_envs.cuda_data_manager.is_data_on_device(
                state
            ), f"{state} is not a valid array name on the GPU!"
            # Note: Discard the first dimension, which is the env dimension
            array_shape = self.cuda_envs.cuda_data_manager.get_shape(state)[1:]

            # Initialize the episode states
            episode_states[state] = np.nan * np.stack(
                [np.ones(array_shape) for _ in range(env.episode_length + 1)]
            )

        for timestep in range(env.episode_length + 1):
            # Update the episode states
            for state in list_of_states:
                episode_states[state][
                    timestep
                ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                    env_id
                ]
            # Evaluate policies to compute action probabilities
            probabilities = self._evaluate_policies()

            # Sample actions
            self._sample_actions(probabilities)

            # Step through all the environments
            self.cuda_envs.step_all_envs()

            # Fetch the states when episode is complete
            if env.cuda_data_manager.pull_data_from_device("_done_")[env_id]:
                for state in list_of_states:
                    episode_states[state][
                        timestep
                    ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                        env_id
                    ]
                break

        return episode_states

    # APIs to integrate with Pytorch Lightning
    # ----------------------------------------

    def forward(self, batch_index, start_event, end_event):
        """
        Perform a forward pass through the model(s) and step through the environment.
        """

        # Evaluate policies to compute action probabilities
        start_event.record()
        probabilities = self._evaluate_policies(batch_index)
        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.policy_eval_time += start_event.elapsed_time(end_event) / 1000

        # Sample actions using the computed probabilities
        # and push to the batch of actions
        start_event.record()
        self._sample_actions(probabilities, batch_index=batch_index)
        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.action_sample_time += start_event.elapsed_time(end_event) / 1000

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

    def _generate_training_batch(self):
        """
        Contains the logic for gathering trajectory data
        to train policy and value network.
        Yields:
           For each policy, a tuple containing actions,
           rewards, done, probs and value function tensors
        """
        # Code timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for batch_index in range(self.training_batch_size_per_env):
            # Evaluate policies and run step functions
            self.forward(batch_index, start_event, end_event)

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
        training_batch = {}
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
            # probabilities_batch, value_functions_batch = self.models[policy](
            #     obs=processed_obs_batch
            # )

            dataset = TensorDataset(
                actions_batch, rewards_batch, done_flags_batch, processed_obs_batch
            )
            training_batch.update(
                {
                    policy: DataLoader(
                        dataset, batch_size=self.training_batch_size_per_env
                    )
                }
            )

        return training_batch

    def train_dataloader(self):
        """
        Get the train data loader.
        """
        return self._generate_training_batch()

    def configure_optimizers(self):
        # Define the optimizers and learning rate schedules
        self.optimizers =[]
        self.lr_schedules = []

        for policy in self.policies:
            # Initialize the (ADAM) optimizer
            policy_config = self._get_config(["policy", policy])
            self.lr_schedules += [ParamScheduler(policy_config["lr"])]

            initial_lr = self.lr_schedules[policy].get_param_value(
                timestep=self.current_timestep[policy]
            )
            self.optimizers += [torch.optim.Adam(
                self.models[policy].parameters(), lr=initial_lr
            )]
        return self.optimizers

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]):
        self.perf_stats.iters += 1
        iteration = self.perf_stats.iters
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        metrics_dict = {}

        # Flag for logging (which also happens after the last iteration)
        logging_flag = (
            iteration % self.config["saving"]["metrics_log_freq"] == 0
            or iteration == self.num_iters - 1
        )

        for policy in batch:
            actions_batch, rewards_batch, done_flags_batch, processed_obs_batch = batch[
                policy
            ]

            probabilities_batch, value_functions_batch = self.models[policy](
                obs=processed_obs_batch
            )

            # Loss and metrics computation
            loss, metrics = self.trainers[policy].compute_loss_and_metrics(
                self.current_timestep[policy],
                actions_batch,
                rewards_batch,
                done_flags_batch,
                probabilities_batch,
                value_functions_batch,
                perform_logging=logging_flag,
            )

            # Compute the gradient norm
            grad_norm = 0.0
            for param in list(
                filter(lambda p: p.grad is not None, self.models[policy].parameters())
            ):
                grad_norm += param.grad.data.norm(2).item()

            # Update the timestep and learning rate based on the schedule
            self.current_timestep[policy] += self.training_batch_size
            lr = self.lr_schedules[policy].get_param_value(
                self.current_timestep[policy]
            )
            for param_group in self.optimizers[policy].param_groups:
                param_group["lr"] = lr

            # Loss backpropagation and optimization step
            self.optimizers[policy].zero_grad()
            loss.backward()

            if self.clip_grad_norm[policy]:
                nn.utils.clip_grad_norm_(
                    self.models[policy].parameters(), self.max_grad_norm[policy]
                )

            self.optimizers[policy].step()

            # Logging
            if logging_flag:
                metrics_dict[policy] = metrics
                # Update the metrics dictionary
                metrics_dict[policy].update(
                    {
                        "Current timestep": self.current_timestep[policy],
                        "Gradient norm": grad_norm,
                        "Learning rate": lr,
                        "Mean episodic reward": self.episodic_reward_sum[policy].item()
                        / (self.num_completed_episodes[policy] + _EPSILON),
                    }
                )
                self.avg_reward[policy] = self.episodic_reward_sum[policy].item() / (
                    self.num_completed_episodes[policy] + _EPSILON
                )

                if self.avg_reward[policy] != 0:
                    print(policy, self.avg_reward[policy])
                # Reset sum and counter
                self.episodic_reward_sum[policy] = (
                    torch.tensor(0).type(torch.float32).cuda()
                )
                self.num_completed_episodes[policy] = 0

        end_event.record()
        torch.cuda.synchronize()

        self.perf_stats.training_time += start_event.elapsed_time(end_event) / 1000

        return OrderedDict(
            {
                "loss": loss,
                "avg_reward": self.avg_reward[policy],
                "log": metrics_dict,
                "progress_bar": metrics_dict,
            }
        )

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        parser.add_argument("--env", type=str, default="CartPole-v0")
        parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
        parser.add_argument(
            "--lam", type=float, default=0.95, help="advantage discount factor"
        )
        parser.add_argument(
            "--lr_actor",
            type=float,
            default=3e-4,
            help="learning rate of actor network",
        )
        parser.add_argument(
            "--lr_critic",
            type=float,
            default=1e-3,
            help="learning rate of critic network",
        )
        parser.add_argument(
            "--max_episode_len",
            type=int,
            default=1000,
            help="capacity of the replay buffer",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=512,
            help="batch_size when training network",
        )
        parser.add_argument(
            "--steps_per_epoch",
            type=int,
            default=2048,
            help="how many action-state pairs to rollout for "
            "trajectory collection per epoch",
        )
        parser.add_argument(
            "--nb_optim_iters",
            type=int,
            default=4,
            help="how many steps of gradient descent to perform on each batch",
        )
        parser.add_argument(
            "--clip_ratio",
            type=float,
            default=0.2,
            help="hyperparameter for clipping in the policy objective",
        )

        return parser


class PerfStats:
    """
    Performance stats that will be included in rollout metrics.
    """

    def __init__(self):
        self.iters = 0
        self.steps = 0
        self.policy_eval_time = 0.0
        self.action_sample_time = 0.0
        self.env_step_time = 0.0
        self.training_time = 0.0
        self.total_time = 0.0

    def get_perf_stats(self):
        return {
            "Mean policy eval time per iter (ms)": self.policy_eval_time
            * 1000
            / self.iters,
            "Mean action sample time per iter (ms)": self.action_sample_time
            * 1000
            / self.iters,
            "Mean env. step time per iter (ms)": self.env_step_time * 1000 / self.iters,
            "Mean training time per iter (ms)": self.training_time * 1000 / self.iters,
            "Mean total time per iter (ms)": self.total_time * 1000 / self.iters,
            "Mean steps per sec (policy eval)": self.steps / self.policy_eval_time,
            "Mean steps per sec (action sample)": self.steps / self.action_sample_time,
            "Mean steps per sec (env. step)": self.steps / self.env_step_time,
            "Mean steps per sec (training time)": self.steps / self.training_time,
            "Mean steps per sec (total)": self.steps / self.total_time,
        }

    def pretty_print(self, stats):
        print("=" * 40)
        print("Speed performance stats")
        print("=" * 40)
        for k, v in stats.items():
            print(f"{k:40}: {v:10.2f}")


class Metrics:
    """
    Metrics class to log and print the key metrics
    """

    def __init__(self):
        pass

    def pretty_print(self, metrics):
        assert metrics is not None
        assert isinstance(metrics, dict)

        for policy in metrics:
            print("=" * 40)
            print(f"Metrics for policy '{policy}'")
            print("=" * 40)
            for k, v in metrics[policy].items():
                print(f"{k:40}: {v:10.5f}")
