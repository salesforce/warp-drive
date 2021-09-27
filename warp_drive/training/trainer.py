# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The Trainer, PerfStats and Metrics classes
"""

import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from gym.spaces import Discrete, MultiDiscrete

from warp_drive.managers.function_manager import CUDASampler
from warp_drive.training.algorithms.a2c import A2C
from warp_drive.training.algorithms.ppo import PPO
from warp_drive.training.models.fully_connected import FullyConnected
from warp_drive.utils.constants import Constants

_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_DONE_FLAGS = Constants.DONE_FLAGS
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_COMBINED = "combined"


def all_equal(iterable):
    """
    Check all elements of an iterable (e.g., list) are identical
    """
    return len(set(iterable)) <= 1


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
        self.config = config

        # Note: experiment name reflects run create time.
        experiment_name = f"{time.time():10.0f}"

        # Checkpoint directory to save model parameters
        self.ckpt_dir = os.path.join(
            self.config["saving"]["basedir"],
            self.config["name"],
            self.config["saving"]["tag"],
            experiment_name,
        )

        # Policies
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.policies = list(config["policy"])
        self.policies_to_train = [
            policy
            for policy in config["policy"]
            if config["policy"][policy]["to_train"]
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
        self.num_episodes = config["trainer"]["num_episodes"]
        self.training_batch_size = config["trainer"]["train_batch_size"]
        self.num_envs = config["trainer"]["num_envs"]

        self.training_batch_size_per_env = self.training_batch_size // self.num_envs
        assert self.training_batch_size_per_env > 0
        self.total_steps = self.cuda_envs.episode_length * self.num_episodes
        self.num_iters = self.total_steps // self.training_batch_size

        self.block = self.cuda_envs.cuda_function_manager.block
        self.data_manager = self.cuda_envs.cuda_data_manager
        self.function_manager = self.cuda_envs.cuda_function_manager

        # Seeding
        seed = config["trainer"].get("seed", np.int32(time.time()))
        self.cuda_sample_controller = CUDASampler(self.cuda_envs.cuda_function_manager)
        self.cuda_sample_controller.init_random(seed)

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.entropy_coeff = config["trainer"]["entropy_coeff"]
        self.algorithm = self.config["trainer"]["algorithm"]
        assert self.algorithm in ["A2C", "PPO"]
        self.vf_loss_coeff = self.config["trainer"]["vf_loss_coeff"]
        self.clip_grad_norm = self.config["trainer"]["clip_grad_norm"]
        if self.clip_grad_norm:
            self.max_grad_norm = self.config["trainer"]["max_grad_norm"]
        self.normalize_advantage = self.config["trainer"]["normalize_advantage"]
        self.normalize_return = self.config["trainer"]["normalize_return"]
        if self.algorithm == "PPO":
            self.clip_param = self.config["trainer"]["clip_param"]

        # Define models and optimizers
        self.models = {}
        self.optimizers = {}
        self.gammas = {}

        for policy in self.policies:
            policy_config = self.config["policy"][policy]
            if policy_config["name"] == "fully_connected":
                self.models[policy] = FullyConnected(
                    env=self.cuda_envs,
                    model_config=policy_config["model"],
                    policy=policy,
                    policy_tag_to_agent_id_map=self.policy_tag_to_agent_id_map,
                    create_separate_placeholders_for_each_policy=create_separate_placeholders_for_each_policy,
                    obs_dim_corresponding_to_num_agents=obs_dim_corresponding_to_num_agents,
                )
            else:
                raise NotImplementedError

            # Load the model parameters (if model checkpoints are specified)
            self.load_model_checkpoint(policy)

            self.models[policy].cuda()
            self.optimizers[policy] = torch.optim.Adam(
                self.models[policy].parameters(), lr=policy_config["lr"]
            )
            self.gammas[policy] = policy_config["gamma"]

        # Performance Stats
        self.perf_stats = PerfStats()
        self.perf_stats.num_iters = self.num_iters
        self.perf_stats.total_steps = self.total_steps

        # Metrics
        self.metrics = Metrics()

    def graceful_close(self):
        # sample controller has random seeds defined in the CUDA memory heap
        del self.cuda_sample_controller
        print("Trainer exits gracefully")

    def generate_rollout(self, batch_index):
        """
        Rollout a single step of the environment.
        """
        assert isinstance(batch_index, int)
        # Code timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Evaluate policy to compute action probabilities and value functions
        start_event.record()
        probabilities = self.evaluate_policies(batch_index=batch_index)
        end_event.record()
        torch.cuda.synchronize()

        self.perf_stats.policy_eval_time += start_event.elapsed_time(end_event) / 1000

        # Sample actions and push to device
        start_event.record()
        self.sample_actions(probabilities, batch_index=batch_index)
        end_event.record()
        torch.cuda.synchronize()

        self.perf_stats.action_sample_time += start_event.elapsed_time(end_event) / 1000

        # Step through all the envs
        start_event.record()
        _, _, done, _ = self.cuda_envs.step_all_envs()

        # Push done flags to done_flags_batch
        done_flags = done["__all__"]
        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            name=f"{_DONE_FLAGS}_batch"
        )[batch_index] = done_flags

        # Push rewards to rewards_batch
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_{policy}"
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_REWARDS}_batch_{policy}"
                )[batch_index] = rewards
        else:
            rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                _REWARDS
            )
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                name=f"{_REWARDS}_batch"
            )[batch_index] = rewards

        # Compute the number of runners left (at episode end).
        if done_flags.any():
            # Reset the env that's in done state.
            self.cuda_envs.reset_only_done_envs()

        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.env_step_time += start_event.elapsed_time(end_event) / 1000

    def evaluate_policies(self, batch_index):
        """
        Perform the policy evaluation (forward pass through the models)
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
                for policy in probabilities:
                    agent_to_id_mapping = self.policy_tag_to_agent_id_map[policy]
                    combined_probabilities[action_idx][
                        :, agent_to_id_mapping
                    ] = probabilities[policy][action_idx]

            probabilities = {_COMBINED: combined_probabilities}

        return probabilities

    def sample_actions(self, probabilities, batch_index):
        """
        Sample action probabilities (and push the sampled actions to the device).
        """
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                suffix = f"_{policy}"
                self.sample_actions_helper(
                    probabilities[policy], batch_index, suffix=suffix
                )
        else:
            assert len(probabilities) == 1
            policy = list(probabilities.keys())[0]
            self.sample_actions_helper(probabilities[policy], batch_index)

    def sample_actions_helper(self, probabilities, batch_index, suffix=""):

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
                # Push indexed actions to actions and actions batch
                actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    action_name
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=_ACTIONS + suffix
                )[:, :, action_idx] = actions
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_ACTIONS}_batch" + suffix
                )[batch_index, :, :, action_idx] = actions

    def register_actions(self, action_space, suffix=""):
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
            for action_idx in range(len(action_dim)):
                self.cuda_sample_controller.register_actions(
                    self.cuda_envs.cuda_data_manager,
                    action_name=f"{_ACTIONS}_{action_idx}" + suffix,
                    num_actions=action_dim[action_idx],
                )

    def train(self):
        """
        Perform training.
        """

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Initialize the trainers
        trainers = {}
        for policy in self.policies_to_train:
            if self.algorithm == "A2C":
                # Advantage Actor-Critic
                trainers[policy] = A2C(
                    discount_factor_gamma=self.gammas[policy],
                    normalize_advantage=self.normalize_advantage,
                    normalize_return=self.normalize_return,
                    vf_loss_coeff=self.vf_loss_coeff,
                    entropy_coeff=self.entropy_coeff,
                )
                print(f"Initializing the A2C trainer for policy {policy}")
            elif self.algorithm == "PPO":
                # Proximal Policy Optimization
                trainers[policy] = PPO(
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

        # Register action placeholders
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                sample_agent_id = self.policy_tag_to_agent_id_map[policy][0]
                action_space = self.cuda_envs.env.action_space[sample_agent_id]

                self.register_actions(action_space, suffix=f"_{policy}")
        else:
            sample_policy = self.policies[0]
            sample_agent_id = self.policy_tag_to_agent_id_map[sample_policy][0]
            action_space = self.cuda_envs.env.action_space[sample_agent_id]

            self.register_actions(action_space)

        # Ensure env is reset before the start of training, and done flags are False
        self.cuda_envs.reset_all_envs()

        for iteration in range(self.num_iters):
            start_time = time.time()

            for batch_index in range(self.training_batch_size_per_env):

                # Generate a rollout for every CUDA environment
                self.generate_rollout(batch_index)

            # Training
            start_event.record()

            metrics = {}

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

            done_flags_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_DONE_FLAGS}_batch"
                )
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
                loss, metrics[policy] = trainers[policy].compute_loss_and_metrics(
                    actions_batch,
                    rewards_batch,
                    done_flags_batch,
                    probabilities_batch,
                    value_functions_batch,
                )

                # Compute the gradient norm
                grad_norm = 0.0
                for p in list(
                    filter(
                        lambda p: p.grad is not None, self.models[policy].parameters()
                    )
                ):
                    grad_norm += p.grad.data.norm(2).item()
                metrics[policy].update({"Gradient norm": grad_norm})

                # Loss backpropagation and optimization step
                self.optimizers[policy].zero_grad()
                loss.backward()

                if self.clip_grad_norm:
                    nn.utils.clip_grad_norm_(
                        self.models[policy].parameters(), self.max_grad_norm
                    )

                self.optimizers[policy].step()

            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.training_time += start_event.elapsed_time(end_event) / 1000

            self.perf_stats.iters += 1

            end_time = time.time()
            self.perf_stats.total_time += end_time - start_time
            self.perf_stats.steps += self.training_batch_size

            if iteration % self.config["saving"]["print_metrics_freq"] == 0:
                self.perf_stats.pretty_print()
                self.metrics.pretty_print(metrics)
                print("\n", "*" * 100, "\n")

            if iteration % self.config["saving"]["save_model_params_freq"] == 0:
                # Save torch model
                self.save_model_checkpoint()

    def load_model_checkpoint(self, policy):
        """
        Load the model parameters if a checkpoint path is specified.
        """
        filepath = self.config["policy"][policy]["model"]["model_ckpt_filepath"]
        if filepath != "":
            assert os.path.isfile(filepath), "Invalid model checkpoint path!"
            print(
                f"Loading the {policy} torch model "
                f"from the previously saved checkpoint: {filepath}"
            )
            self.models[policy].load_state_dict(torch.load(filepath))

    def save_model_checkpoint(self):
        """
        Save the model parameters
        """
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=False)

        for policy in self.models:
            filepath = os.path.join(
                self.ckpt_dir, f"{policy}_{self.perf_stats.steps}.state_dict"
            )
            print(f"Saving the {policy} torch model to the file: {filepath}.")

            torch.save(self.models[policy].state_dict(), filepath)

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
            probabilities = self.evaluate_policies(batch_index=t)

            # Sample actions and push to device
            self.sample_actions(probabilities, batch_index=t)

            # Step through all the environments
            _, _, done, _ = self.cuda_envs.step_all_envs()

            # Update the global states
            for state in list_of_states:
                global_states[state][
                    t + 1
                ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                    env_id
                ]

            # Fetch the global states when episode is complete
            if env.cuda_data_manager.pull_data_from_device("_done_")[env_id]:
                return {state: global_states[state][: t + 2] for state in global_states}


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

    def get(self):
        return {
            "mean_policy_eval_time_per_iter (ms)": self.policy_eval_time
            * 1000
            / self.iters,
            "mean_action_sample_time_per_iter (ms)": self.action_sample_time
            * 1000
            / self.iters,
            "mean_env_step_time_per_iter (ms)": self.env_step_time * 1000 / self.iters,
            "mean_training_time_per_iter (ms)": self.training_time * 1000 / self.iters,
            "mean_total_time_per_iter (ms)": self.total_time * 1000 / self.iters,
            "mean_steps_per_sec (policy_eval)": self.steps / self.policy_eval_time,
            "mean_steps_per_sec (action_sample)": self.steps / self.action_sample_time,
            "mean_steps_per_sec (env_step)": self.steps / self.env_step_time,
            "mean_steps_per_sec (training_time)": self.steps / self.training_time,
            "mean_steps_per_sec (total)": self.steps / self.total_time,
        }

    def pretty_print(self):
        stats = self.get()
        print("\n")
        print("=" * 20)
        print("Performance Stats")
        print("=" * 20)
        print(f"{'iterations completed':40}: {self.iters} / {self.num_iters}")
        print(f"{'steps completed':40}: {self.steps} / {self.total_steps}")
        [print(f"{k:40}: {v:10.2f}") for k, v in stats.items()]


class Metrics:
    """
    Metrics class to print the key metrics
    """

    def __init__(self):
        self.iters = 0
        self.steps = 0

    def pretty_print(self, metrics):
        assert metrics is not None
        assert isinstance(metrics, dict)

        print("\n")
        for policy in metrics:
            print("=" * 40)
            print(f"Metrics for policy '{policy}'")
            print("=" * 40)
            [print(f"{k:40}: {v:10.5f}") for k, v in metrics[policy].items()]
