# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The Trainer, PerfStats and Metrics classes
"""

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

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_DONE_FLAGS = Constants.DONE_FLAGS


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
    ):
        assert env_wrapper is not None
        assert config is not None
        assert isinstance(policy_tag_to_agent_id_map, dict)
        assert len(policy_tag_to_agent_id_map) > 0  # at least one policy

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
        self.only_one_policy = len(self.policies) == 1
        # if only one policy and discrete action, then we can use sampler
        # to sample and write to actions directly
        self.fast_sampling_mode = False

        # Number of iterations algebra
        self.num_episodes = config["trainer"]["num_episodes"]
        self.training_batch_size = config["trainer"]["train_batch_size"]
        self.num_envs = config["trainer"]["num_envs"]

        self.training_batch_size_per_env = self.training_batch_size // self.num_envs
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
                )
            else:
                raise NotImplementedError
            if (
                hasattr(self.models[policy], "set_fast_forward_mode")
                and self.only_one_policy is True
            ):
                self.models[policy].set_fast_forward_mode()

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
        # Code timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Policy evaluation to compute action probabilities
        start_event.record()

        probabilities = {}
        for policy in self.policies:
            probabilities[policy], _ = self.models[policy](
                batch_index=batch_index,
                batch_size=self.training_batch_size_per_env,
            )

        end_event.record()
        torch.cuda.synchronize()

        self.perf_stats.policy_eval_time += start_event.elapsed_time(end_event) / 1000

        # Action sampling
        start_event.record()

        if self.fast_sampling_mode:
            probs = probabilities[self.policies[0]][0]
            self.cuda_sample_controller.sample(
                self.cuda_envs.cuda_data_manager, probs, _ACTIONS
            )
            actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                _ACTIONS
            )
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                name=f"{_ACTIONS}_batch"
            )[batch_index] = actions

        else:
            for policy in self.policies:
                actions_dict = {}
                for idx, probs in enumerate(probabilities[policy]):
                    self.cuda_sample_controller.sample(
                        self.cuda_envs.cuda_data_manager,
                        probs,
                        f"{_ACTIONS}_{policy}_{idx}",
                    )
                    actions_dict[
                        idx
                    ] = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"{_ACTIONS}_{policy}_{idx}"
                    )

                # Push actions to actions_batch
                num_actions = len(probabilities[policy])
                if num_actions == 1:  # Discrete actions
                    actions = actions_dict[0]
                elif num_actions > 1:  # MultiDiscrete actions
                    actions = torch.stack(
                        [actions_dict[idx] for idx in range(num_actions)],
                        dim=-1,
                    )

                agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=_ACTIONS
                )[:, agent_ids_for_policy, :] = actions

                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_ACTIONS}_batch"
                )[batch_index, :, agent_ids_for_policy, :] = actions

        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.action_sample_time += start_event.elapsed_time(end_event) / 1000

        # Step through the env
        start_event.record()
        _, _, done, _ = self.cuda_envs.step()

        # Push done flags to done_flags_batch
        done_flags = done["__all__"]
        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            name=f"{_DONE_FLAGS}_batch"
        )[batch_index] = done_flags

        # Push rewards to rewards_batch
        rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(_REWARDS)
        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            name=f"{_REWARDS}_batch"
        )[batch_index] = rewards

        # Compute the number of runners left (at episode end)
        if done_flags.any():
            # Reset the env that's in done state
            self.cuda_envs.reset_only_done_envs()

        end_event.record()
        torch.cuda.synchronize()
        self.perf_stats.env_step_time += start_event.elapsed_time(end_event) / 1000

    def train(self):
        """
        Perform training.
        """

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Register action placeholders for each policy
        for policy in self.policies:
            sample_agent_id = self.policy_tag_to_agent_id_map[policy][0]
            action_space = self.cuda_envs.env.action_space[sample_agent_id]

            if isinstance(action_space, Discrete):
                # Single action
                if not self.only_one_policy:
                    self.cuda_sample_controller.register_actions(
                        self.cuda_envs.cuda_data_manager,
                        action_name=f"{_ACTIONS}_{policy}_0",
                        num_actions=action_space.n,
                    )
                else:
                    # if only one policy and this policy has only a discrete action
                    self.fast_sampling_mode = True
                    print(
                        "the trainer turns on the fast_sampling_mode to speed up "
                        "the action sampling (there is only one policy with discrete "
                        "action space, therefore there is no need to have "
                        "torch gpu concatenates multiple actions samplings "
                        "and map to agents which is slow) "
                    )
                    self.cuda_sample_controller.register_actions(
                        self.cuda_envs.cuda_data_manager,
                        action_name=_ACTIONS,
                        num_actions=action_space.n,
                    )
            elif isinstance(action_space, MultiDiscrete):
                # Multiple actions
                for idx in range(len(action_space.nvec)):
                    self.cuda_sample_controller.register_actions(
                        self.cuda_envs.cuda_data_manager,
                        action_name=f"{_ACTIONS}_{policy}_{idx}",
                        num_actions=action_space.nvec[idx],
                    )
            else:
                raise NotImplementedError(
                    "Action spaces can be of type" "Discrete or MultiDiscrete"
                )

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
                print("Initializing the A2C trainer")
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
                print("Initializing the PPO trainer")
            else:
                raise NotImplementedError

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
            full_obs_batch = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                f"{_OBSERVATIONS}_batch"
            )
            full_actions_batch = (
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_ACTIONS}_batch"
                )
            )
            full_rewards_batch = (
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

                agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                obs_batch = full_obs_batch[:, :, agent_ids_for_policy, :]
                actions_batch = full_actions_batch[:, :, agent_ids_for_policy, :]
                rewards_batch = full_rewards_batch[:, :, agent_ids_for_policy]

                probabilities_batch, value_function_batch = self.models[policy](
                    obs=obs_batch
                )

                grad_norm = 0.0
                for p in list(
                    filter(
                        lambda p: p.grad is not None, self.models[policy].parameters()
                    )
                ):
                    grad_norm += p.grad.data.norm(2).item()

                loss, metrics[policy] = trainers[policy].compute_loss_and_metrics(
                    actions_batch,
                    rewards_batch,
                    done_flags_batch,
                    probabilities_batch,
                    value_function_batch,
                )

                metrics[policy].update({"Gradient norm": grad_norm})

                # Optimization step
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
        list_of_states=None,
    ):
        """
        Step through env and fetch the global states for an episode
        """
        assert list_of_states is not None
        assert isinstance(list_of_states, list)
        assert len(list_of_states) > 0

        self.cuda_envs.reset_all_envs()
        env = self.cuda_envs.env

        global_states = {}
        # Just use the data from the first env
        env_id = 0

        for state in list_of_states:
            global_states[state] = np.zeros((env.episode_length + 1, env.num_agents))
            global_states[state][
                0
            ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[env_id]

        for t in range(1, env.episode_length + 1):
            obs = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                _OBSERVATIONS
            )
            for policy in self.policies:
                obs_policy = obs[:, self.policy_tag_to_agent_id_map[policy]]
                probabilities, _ = self.models[policy](obs=obs_policy)
                actions_dict = {}
                for idx, probs in enumerate(probabilities):
                    self.cuda_sample_controller.sample(
                        self.cuda_envs.cuda_data_manager,
                        probs,
                        f"sampled_actions_{policy}_{idx}",
                    )
                    actions_dict[
                        idx
                    ] = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        f"sampled_actions_{policy}_{idx}"
                    )

                # Push actions
                actions = torch.stack(
                    [actions_dict[key] for key in range(len(actions_dict))], dim=-1
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name="sampled_actions"
                )[:, self.policy_tag_to_agent_id_map[policy], :] = actions

            self.cuda_envs.step()

            # Update the global states
            for state in list_of_states:
                global_states[state][
                    t
                ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                    env_id
                ]

            # Fetch the global states when episode is complete
            if env.cuda_data_manager.pull_data_from_device("_done_")[env_id]:
                return {state: global_states[state][: t + 1] for state in global_states}


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
