# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The Trainer, PerfStats and Metrics classes
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
from torch.nn.parallel import DistributedDataParallel as DDP
from warp_drive.training.trainer_base import TrainerBase, all_equal, verbose_print

from warp_drive.training.algorithms.policygradient.a2c import A2C
from warp_drive.training.algorithms.policygradient.ppo import PPO
from warp_drive.training.models.factory import ModelFactory
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


class TrainerA2C(TrainerBase):
    def __init__(
        self,
        env_wrapper=None,
        config=None,
        policy_tag_to_agent_id_map=None,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
        num_devices=1,
        device_id=0,
        results_dir=None,
        verbose=True,
    ):
        self.models = {}
        super().__init__(
            env_wrapper=env_wrapper,
            config=config,
            policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=create_separate_placeholders_for_each_policy,
            obs_dim_corresponding_to_num_agents=obs_dim_corresponding_to_num_agents,
            num_devices=num_devices,
            device_id=device_id,
            results_dir=results_dir,
            verbose=verbose,
        )

    def _initialize_policy_algorithm(self, policy):
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
        gamma = self._get_config(["policy", policy, "gamma"])
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
            clip_param = self._get_config(["policy", policy, "clip_param"])
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

    def _initialize_policy_model(self, policy):
        policy_model_config = self._get_config(["policy", policy, "model"])
        model_obj = ModelFactory.create(policy_model_config["type"])
        model = model_obj(
            env=self.cuda_envs,
            model_config=policy_model_config,
            policy=policy,
            policy_tag_to_agent_id_map=self.policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=self.create_separate_placeholders_for_each_policy,
            obs_dim_corresponding_to_num_agents=self.obs_dim_corresponding_to_num_agents,
        )

        if "init_method" in policy_model_config and \
                policy_model_config["init_method"] == "xavier":
            def init_weights_by_xavier_uniform(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight)

            model.apply(init_weights_by_xavier_uniform)

        self.models[policy] = model

    def _send_policy_model_to_device(self, policy):
        self.models[policy].cuda()
        # If distributed train, sync model using DDP
        if self.num_devices > 1:
            self.models[policy] = DDP(
                self.models[policy], device_ids=[self.device_id]
            )
            self.ddp_mode[policy] = True
        else:
            self.ddp_mode[policy] = False

    def _initialize_optimizer(self, policy):
        # Initialize the (ADAM) optimizer
        lr_config = self._get_config(["policy", policy, "lr"])
        self.lr_schedules[policy] = ParamScheduler(lr_config)
        initial_lr = self.lr_schedules[policy].get_param_value(
            timestep=self.current_timestep[policy]
        )
        self.optimizers[policy] = torch.optim.Adam(
            self.models[policy].parameters(), lr=initial_lr
        )

    def _evaluate_policies(self, batch_index=0):
        """
        Perform the policy evaluation (forward pass through the models)
        and compute action probabilities
        """
        assert isinstance(batch_index, int)
        probabilities = {}
        for policy in self.policies:
            if self.ddp_mode[policy]:
                # self.models[policy] is a DDP wrapper of the model instance
                obs = self.models[policy].module.process_one_step_obs()
                self.models[policy].module.push_processed_obs_to_batch(batch_index, obs)
            else:
                obs = self.models[policy].process_one_step_obs()
                self.models[policy].push_processed_obs_to_batch(batch_index, obs)
            probabilities[policy], _ = self.models[policy](obs)

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

            first_action_type_id = 0
            num_envs = probabilities[first_policy][first_action_type_id].shape[0]
            num_agents = self.cuda_envs.env.num_agents

            combined_probabilities = [None for _ in range(num_action_types)]
            for action_type_id in range(num_action_types):
                action_dim = probabilities[first_policy][action_type_id].shape[-1]
                combined_probabilities[action_type_id] = torch.zeros(
                    (num_envs, num_agents, action_dim)
                ).cuda()

            # Combine the probabilities across policies
            for action_type_id in range(num_action_types):
                for policy, prob_values in probabilities.items():
                    agent_to_id_mapping = self.policy_tag_to_agent_id_map[policy]
                    combined_probabilities[action_type_id][
                        :, agent_to_id_mapping
                    ] = prob_values[action_type_id]

            probabilities = {_COMBINED: combined_probabilities}

        return probabilities

    def _update_model_params(self, iteration):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Flag for logging (which also happens after the last iteration)
        logging_flag = (
            iteration % self.config["saving"]["metrics_log_freq"] == 0
            or iteration == self.num_iters - 1
        )

        metrics_dict = {}

        done_flags_batch = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
            f"{_DONE_FLAGS}_batch"
        )
        # On the device, observations_batch, actions_batch,
        # rewards_batch are all shaped
        # (batch_size, num_envs, num_agents, *feature_dim).
        # done_flags_batch is shaped (batch_size, num_envs)
        # Perform training sequentially for each policy
        for policy in self.policies_to_train:
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
                        "Mean episodic steps": self.episodic_step_sum[policy].item()
                        / (self.num_completed_episodes[policy] + _EPSILON),
                    }
                )

                # Reset sum and counter
                self.episodic_reward_sum[policy] = (
                    torch.tensor(0).type(torch.float32).cuda()
                )
                self.episodic_step_sum[policy] = (
                    torch.tensor(0).type(torch.int64).cuda()
                )
                self.num_completed_episodes[policy] = 0

        end_event.record()
        torch.cuda.synchronize()

        self.perf_stats.training_time += start_event.elapsed_time(end_event) / 1000
        return metrics_dict

    def _load_model_checkpoint_helper(self, policy, ckpt_filepath):
        if ckpt_filepath != "":
            assert os.path.isfile(ckpt_filepath), "Invalid model checkpoint path!"
            if self.verbose:
                verbose_print(
                    f"Loading the '{policy}' torch model "
                    f"from the previously saved checkpoint: '{ckpt_filepath}'",
                    self.device_id,
                )
            self.models[policy].load_state_dict(torch.load(ckpt_filepath))

            # Update the current timestep using the saved checkpoint filename
            timestep = int(ckpt_filepath.split(".state_dict")[0].split("_")[-1])
            if self.verbose:
                verbose_print(
                    f"Updating the timestep for the '{policy}' model to {timestep}.",
                    self.device_id,
                )
            self.current_timestep[policy] = timestep

    def save_model_checkpoint(self, iteration=0):
        """
        Save the model parameters
        """
        # If multiple devices, save the synced-up model only for device id 0
        if self.device_id == 0:
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
                    if self.verbose:
                        verbose_print(
                            f"Saving the '{policy}' torch model "
                            f"to the file: '{filepath}'.",
                            self.device_id,
                        )

                    torch.save(model.state_dict(), filepath)
