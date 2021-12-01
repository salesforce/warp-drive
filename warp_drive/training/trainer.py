# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The Trainer, PerfStats and Metrics classes
"""

import json
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


class Trainer:
    """
    The trainer object. Contains modules train(), save_model_checkpoint() and
    fetch_episode_global_states()
    """

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
            self.get_config(["saving", "basedir"]),
            self.get_config(["saving", "name"]),
            self.get_config(["saving", "tag"]),
            experiment_name,
        )
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=False)

        # Policies
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.policies = list(self.get_config(["policy"]).keys())
        self.policies_to_train = [
            policy
            for policy in self.policies
            if self.get_config(["policy", policy, "to_train"])
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
        self.num_episodes = self.get_config(["trainer", "num_episodes"])
        self.training_batch_size = self.get_config(["trainer", "train_batch_size"])
        self.num_envs = self.get_config(["trainer", "num_envs"])

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

        self.entropy_coeff = self.get_config(["trainer", "entropy_coeff"])
        self.algorithm = self.get_config(["trainer", "algorithm"])
        assert self.algorithm in ["A2C", "PPO"]
        self.vf_loss_coeff = self.get_config(["trainer", "vf_loss_coeff"])
        self.clip_grad_norm = self.get_config(["trainer", "clip_grad_norm"])
        if self.clip_grad_norm:
            self.max_grad_norm = self.get_config(["trainer", "max_grad_norm"])
        self.normalize_advantage = self.get_config(["trainer", "normalize_advantage"])
        self.normalize_return = self.get_config(["trainer", "normalize_return"])
        if self.algorithm == "PPO":
            self.clip_param = self.get_config(["trainer", "clip_param"])

        # Define models and optimizers
        self.models = {}
        self.optimizers = {}
        self.lr_schedules = {}
        self.gammas = {}

        # For logging episodic reward
        self.num_completed_episodes = {}
        self.episodic_reward_sum = {}
        self.reward_running_sum = {}

        # Indicates the current timestep of the policy model
        self.current_timestep = {}

        for policy in self.policies:
            self.current_timestep[policy] = 0
            policy_config = self.get_config(["policy", policy])
            if policy_config["name"] == "fully_connected":
                self.models[policy] = FullyConnected(
                    self.cuda_envs,
                    policy_config["model"]["fc_dims"],
                    policy,
                    self.policy_tag_to_agent_id_map,
                    create_separate_placeholders_for_each_policy,
                    obs_dim_corresponding_to_num_agents,
                )
            else:
                raise NotImplementedError

            # Load the model parameters (if model checkpoints are specified)
            # Note: Loading the model checkpoint may also update the current timestep!
            self.load_model_checkpoint(policy)

            self.total_steps = self.cuda_envs.episode_length * self.num_episodes
            self.num_iters = self.total_steps // self.training_batch_size

            # Push the models to the GPU
            self.models[policy].cuda()

            # Initialize the (ADAM) optimizer
            self.lr_schedules[policy] = ParamScheduler(policy_config["lr"])
            initial_lr = self.lr_schedules[policy].get_param_value(
                timestep=self.current_timestep[policy]
            )
            self.optimizers[policy] = torch.optim.Adam(
                self.models[policy].parameters(), lr=initial_lr
            )

            self.gammas[policy] = policy_config["gamma"]

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
        for policy in self.policies_to_train:
            if self.algorithm == "A2C":
                # Advantage Actor-Critic
                self.trainers[policy] = A2C(
                    discount_factor_gamma=self.gammas[policy],
                    normalize_advantage=self.normalize_advantage,
                    normalize_return=self.normalize_return,
                    vf_loss_coeff=self.vf_loss_coeff,
                    entropy_coeff=self.entropy_coeff,
                )
                print(f"Initializing the A2C trainer for policy {policy}")
            elif self.algorithm == "PPO":
                # Proximal Policy Optimization
                self.trainers[policy] = PPO(
                    discount_factor_gamma=self.gammas[policy],
                    clip_param=self.clip_param,
                    normalize_advantage=self.normalize_advantage,
                    normalize_return=self.normalize_return,
                    vf_loss_coeff=self.vf_loss_coeff,
                    entropy_coeff=self.entropy_coeff,
                )
                print(f"Initializing the PPO trainer for policy {policy}")
            else:
                raise NotImplementedError

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

    def get_config(self, args):
        assert isinstance(args, (tuple, list))
        config = self.config
        for arg in args:
            try:
                config = config[arg]
            except ValueError:
                print("Missing configuration '{arg}'!")
        return config

    def train(self):
        """
        Perform training.
        """
        # Ensure env is reset before the start of training, and done flags are False
        self.cuda_envs.reset_all_envs()

        for iteration in range(self.num_iters):
            start_time = time.time()

            # Generate a batched rollout for every CUDA environment.
            self.generate_rollout_batch()

            # Train / update model parameters.
            metrics = self.update_model_params(iteration)

            self.perf_stats.iters = iteration + 1
            self.perf_stats.steps = self.perf_stats.iters * self.training_batch_size
            end_time = time.time()
            self.perf_stats.total_time += end_time - start_time

            # Log the training metrics
            self.log_metrics(metrics)

            # Save torch model
            self.save_model_checkpoint(iteration)

    def generate_rollout_batch(self):
        """
        Generate an environment rollout batch.
        """
        # Code timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        for batch_index in range(self.training_batch_size_per_env):

            # Evaluate policies to compute action probabilities
            start_event.record()
            probabilities = self.evaluate_policies(batch_index=batch_index)
            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.policy_eval_time += (
                start_event.elapsed_time(end_event) / 1000
            )

            # Sample actions using the computed probabilities
            # and push to actions batch
            start_event.record()
            self.sample_actions(probabilities, batch_index=batch_index)
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

    def evaluate_policies(self, batch_index):
        """
        Perform the policy evaluation (forward pass through the models)
        and compute action probabilities
        """
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

    def sample_actions(self, probabilities, batch_index):
        """
        Sample action probabilities (and push the sampled actions to the device).
        """
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
            # Push actions to actions batch
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
                # Push (indexed) actions to actions and actions batch
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

    def update_model_params(self, iteration):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Flag for logging (which also happens after the last iteration)
        logging_flag = (
            iteration % self.config["saving"]["metrics_log_freq"] == 0
            or iteration == self.num_iters - 1
        )

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

            if self.clip_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.models[policy].parameters(), self.max_grad_norm
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

                # Reset sum and counter
                self.episodic_reward_sum[policy] = (
                    torch.tensor(0).type(torch.float32).cuda()
                )
                self.num_completed_episodes[policy] = 0

        end_event.record()
        torch.cuda.synchronize()

        self.perf_stats.training_time += start_event.elapsed_time(end_event) / 1000
        return metrics_dict

    def log_metrics(self, metrics):
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

    def load_model_checkpoint(self, policy):
        """
        Load the model parameters if a checkpoint path is specified.
        """
        filepath = self.config["policy"][policy]["model"]["model_ckpt_filepath"]
        if filepath != "":
            assert os.path.isfile(filepath), "Invalid model checkpoint path!"
            print(
                f"Loading the '{policy}' torch model "
                f"from the previously saved checkpoint: '{filepath}'"
            )
            self.models[policy].load_state_dict(torch.load(filepath))

            # Update the current timestep using the saved checkpoint filename
            timestep = int(filepath.split(".state_dict")[0].split("_")[-1])
            print(f"Updating the timestep for the '{policy}' model to {timestep}.")
            self.current_timestep[policy] = timestep

    def save_model_checkpoint(self, iteration):
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

    def fetch_episode_global_states(
        self,
        env_id=0,  # environment id to fetch the states from
        list_of_states=None,  # list of (global) states to fetch
    ):
        """
        Step through env and fetch the global states for an entire episode
        """
        assert 0 <= env_id < self.num_envs
        assert list_of_states is not None
        assert isinstance(list_of_states, list)
        assert len(list_of_states) > 0

        self.cuda_envs.reset_all_envs()
        env = self.cuda_envs.env

        global_states = {}

        for state in list_of_states:
            assert self.cuda_envs.cuda_data_manager.is_data_on_device(state)
            global_states[state] = np.zeros((env.episode_length + 1, env.num_agents))
            global_states[state][
                0
            ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[env_id]

        for t in range(env.episode_length):
            # Evaluate policies to compute action probabilities
            probabilities = self.evaluate_policies(batch_index=t)

            # Sample actions
            self.sample_actions(probabilities, batch_index=t)

            # Step through all the environments
            self.cuda_envs.step_all_envs()

            # Update the global states
            for state in list_of_states:
                global_states[state][
                    t + 1
                ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                    env_id
                ]

            # Fetch the global states when episode is complete
            if env.cuda_data_manager.pull_data_from_device("_done_")[env_id]:
                break

        return {
            key: global_state[: t + 2] for key, global_state in global_states.items()
        }


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
