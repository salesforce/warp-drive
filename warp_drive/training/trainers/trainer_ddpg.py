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
from warp_drive.training.trainers.trainer_base import TrainerBase, all_equal, verbose_print

from warp_drive.training.algorithms.policygradient.ddpg import DDPG
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


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


class TrainerDDPG(TrainerBase):
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
        # Define models, optimizers, and learning rate schedules
        self.actor_models = {}
        self.critic_models = {}
        self.target_actor_models = {}
        self.target_critic_models = {}
        self.actor_optimizers = {}
        self.critic_optimizers = {}
        self.actor_lr_schedules = {}
        self.critic_lr_schedules = {}
        self.tau = {}

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
        assert algorithm in ["DDPG"]
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
        self.tau[policy] = self._get_config(["policy", policy, "tau"])
        if algorithm == "DDPG":
            # Advantage Actor-Critic
            self.trainers[policy] = DDPG(
                discount_factor_gamma=gamma,
                normalize_advantage=normalize_advantage,
                normalize_return=normalize_return,
            )
            logging.info(f"Initializing the DDPG trainer for policy {policy}")
        else:
            raise NotImplementedError

    def _initialize_policy_model(self, policy):
        if not isinstance(self._get_config(["policy", policy, "model"]), dict) or \
                "actor" not in self._get_config(["policy", policy, "model"]) or "critic" \
                not in self._get_config(["policy", policy, "model"]):
            actor_model_config = self._get_config(["policy", policy, "model"])
            critic_model_config = actor_model_config
        else:
            actor_model_config = self._get_config(["policy", policy, "model", "actor"])
            critic_model_config = self._get_config(["policy", policy, "model", "critic"])

        model_obj_actor = ModelFactory.create(actor_model_config["type"])
        actor = model_obj_actor(
            env=self.cuda_envs,
            model_config=actor_model_config,
            policy=policy,
            policy_tag_to_agent_id_map=self.policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=self.create_separate_placeholders_for_each_policy,
            obs_dim_corresponding_to_num_agents=self.obs_dim_corresponding_to_num_agents,
        )
        target_actor = model_obj_actor(
            env=self.cuda_envs,
            model_config=actor_model_config,
            policy=policy,
            policy_tag_to_agent_id_map=self.policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=self.create_separate_placeholders_for_each_policy,
            obs_dim_corresponding_to_num_agents=self.obs_dim_corresponding_to_num_agents,
        )

        if "init_method" in actor_model_config and \
                actor_model_config["init_method"] == "xavier":
            def init_weights_by_xavier_uniform(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight)

            actor.apply(init_weights_by_xavier_uniform)

        hard_update(target_actor, actor)
        self.actor_models[policy] = actor
        self.target_actor_models[policy] = target_actor

        model_obj_critic = ModelFactory.create(critic_model_config["type"])
        critic = model_obj_critic(
            env=self.cuda_envs,
            model_config=critic_model_config,
            policy=policy,
            policy_tag_to_agent_id_map=self.policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=self.create_separate_placeholders_for_each_policy,
            obs_dim_corresponding_to_num_agents=self.obs_dim_corresponding_to_num_agents,
        )
        target_critic = model_obj_critic(
            env=self.cuda_envs,
            model_config=critic_model_config,
            policy=policy,
            policy_tag_to_agent_id_map=self.policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=self.create_separate_placeholders_for_each_policy,
            obs_dim_corresponding_to_num_agents=self.obs_dim_corresponding_to_num_agents,
        )

        if "init_method" in critic_model_config and \
                critic_model_config["init_method"] == "xavier":
            def init_weights_by_xavier_uniform(m):
                if isinstance(m, nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight)

            critic.apply(init_weights_by_xavier_uniform)

        hard_update(target_critic, critic)
        self.critic_models[policy] = critic
        self.target_critic_models[policy] = target_critic

    def _send_policy_model_to_device(self, policy):
        self.actor_models[policy].cuda()
        self.critic_models[policy].cuda()
        self.target_actor_models[policy].cuda()
        self.target_critic_models[policy].cuda()
        # If distributed train, sync model using DDP
        if self.num_devices > 1:
            self.actor_models[policy] = DDP(
                self.actor_models[policy], device_ids=[self.device_id]
            )
            self.critic_models[policy] = DDP(
                self.critic_models[policy], device_ids=[self.device_id]
            )
            self.target_actor_models[policy] = DDP(
                self.target_actor_models[policy], device_ids=[self.device_id]
            )
            self.target_critic_models[policy] = DDP(
                self.target_critic_models[policy], device_ids=[self.device_id]
            )
            self.ddp_mode[policy] = True
        else:
            self.ddp_mode[policy] = False

    def _initialize_optimizer(self, policy):
        # Initialize the (ADAM) optimizer
        if not isinstance(self._get_config(["policy", policy, "lr"]), dict) or \
                "actor" not in self._get_config(["policy", policy, "lr"]) or "critic" \
                not in self._get_config(["policy", policy, "lr"]):
            actor_lr_config = self._get_config(["policy", policy, "lr"])
            critic_lr_config = actor_lr_config
        else:
            actor_lr_config = self._get_config(["policy", policy, "lr", "actor"])
            critic_lr_config = self._get_config(["policy", policy, "lr", "critic"])

        self.actor_lr_schedules[policy] = ParamScheduler(actor_lr_config)
        self.critic_lr_schedules[policy] = ParamScheduler(critic_lr_config)
        initial_actor_lr = self.actor_lr_schedules[policy].get_param_value(
            timestep=self.current_timestep[policy]
        )
        initial_critic_lr = self.critic_lr_schedules[policy].get_param_value(
            timestep=self.current_timestep[policy]
        )
        self.actor_optimizers[policy] = torch.optim.Adam(
            self.actor_models[policy].parameters(), lr=initial_actor_lr
        )
        self.critic_optimizers[policy] = torch.optim.Adam(
            self.critic_models[policy].parameters(), lr=initial_critic_lr
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
                obs = self.actor_models[policy].module.process_one_step_obs()
                self.actor_models[policy].module.push_processed_obs_to_batch(batch_index, obs)
            else:
                obs = self.actor_models[policy].process_one_step_obs()
                self.actor_models[policy].push_processed_obs_to_batch(batch_index, obs)
            probabilities[policy] = self.actor_models[policy](obs)

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
                assert action_dim == 1, "action_dim != 1 but DDPG samples deterministic actions"
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
            probabilities_batch = self.actor_models[policy](
                obs=processed_obs_batch
            )
            target_probabilities_batch = self.target_actor_models[policy](
                obs=processed_obs_batch
            )
            # Critic Q(s_t, a_t) is a function of both obs and action
            # value_functions_batch includes sampled actions
            value_functions_batch = self.critic_models[policy](
                obs=processed_obs_batch, action=actions_batch
            )
            # Critic Q(s_t+1, a_t+1) is a function of both obs and action
            # next_value_functions_batch not includes sampled action but
            # the detached output from actor directly
            next_value_functions_batch = self.target_critic_models[policy](
                obs=processed_obs_batch[1:], action=[pb[1:].detach() for pb in target_probabilities_batch]
            )
            # j_functions_batch includes the graph of actor network for the back-propagation
            j_functions_batch = self.critic_models[policy](
                obs=processed_obs_batch, action=probabilities_batch
            )
            # Loss and metrics computation
            actor_loss, critic_loss, metrics = self.trainers[policy].compute_loss_and_metrics(
                self.current_timestep[policy],
                actions_batch,
                rewards_batch,
                done_flags_batch,
                value_functions_batch,
                next_value_functions_batch,
                j_functions_batch,
                perform_logging=logging_flag,
            )
            # Compute the gradient norm
            actor_grad_norm = 0.0
            for param in list(
                filter(lambda p: p.grad is not None, self.actor_models[policy].parameters())
            ):
                actor_grad_norm += param.grad.data.norm(2).item()

            critic_grad_norm = 0.0
            for param in list(
                    filter(lambda p: p.grad is not None, self.critic_models[policy].parameters())
            ):
                critic_grad_norm += param.grad.data.norm(2).item()

            # Update the timestep and learning rate based on the schedule
            self.current_timestep[policy] += self.training_batch_size
            actor_lr = self.actor_lr_schedules[policy].get_param_value(
                self.current_timestep[policy]
            )
            for param_group in self.actor_optimizers[policy].param_groups:
                param_group["lr"] = actor_lr

            critic_lr = self.critic_lr_schedules[policy].get_param_value(
                self.current_timestep[policy]
            )
            for param_group in self.critic_optimizers[policy].param_groups:
                param_group["lr"] = critic_lr

            # Loss backpropagation and optimization step
            self.actor_optimizers[policy].zero_grad()
            self.critic_optimizers[policy].zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            if self.clip_grad_norm[policy]:
                nn.utils.clip_grad_norm_(
                    self.actor_models[policy].parameters(), self.max_grad_norm[policy]
                )
                nn.utils.clip_grad_norm_(
                    self.critic_models[policy].parameters(), self.max_grad_norm[policy]
                )

            self.actor_optimizers[policy].step()
            self.critic_optimizers[policy].step()

            soft_update(self.target_actor_models[policy], self.actor_models[policy], self.tau[policy])
            soft_update(self.target_critic_models[policy], self.critic_models[policy], self.tau[policy])
            # Logging
            if logging_flag:
                metrics_dict[policy] = metrics
                # Update the metrics dictionary
                metrics_dict[policy].update(
                    {
                        "Current timestep": self.current_timestep[policy],
                        "Gradient norm (Actor)": actor_grad_norm,
                        "Gradient norm (Critic)": critic_grad_norm,
                        "Learning rate (Actor)": actor_lr,
                        "Learning rate (Critic)": critic_lr,
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
        if isinstance(ckpt_filepath, dict) and "actor" in ckpt_filepath and "critic" in ckpt_filepath:
            if ckpt_filepath["actor"] != "" and ckpt_filepath["critic"] != "":
                assert os.path.isfile(ckpt_filepath["actor"]), "Invalid actor model checkpoint path!"
                assert os.path.isfile(ckpt_filepath["critic"]), "Invalid critic model checkpoint path!"
                # Update the current timestep using the saved checkpoint filename
                actor_timestep = int(ckpt_filepath["actor"].split(".state_dict")[0].split("_")[-1])
                critic_timestep = int(ckpt_filepath["critic"].split(".state_dict")[0].split("_")[-1])
                assert actor_timestep == critic_timestep, \
                    "The timestep is different between the actor model and the critic model "
                if self.verbose:
                    verbose_print(
                        f"Loading the '{policy}' torch model "
                        f"from the previously saved checkpoint: "
                        f"actor: '{ckpt_filepath['actor']}'"
                        f"critic: '{ckpt_filepath['critic']}'",
                        self.device_id,
                    )
                self.actor_models[policy].load_state_dict(torch.load(ckpt_filepath["actor"]))
                self.critic_models[policy].load_state_dict(torch.load(ckpt_filepath["critic"]))

                if self.verbose:
                    verbose_print(
                        f"Updating the timestep for the '{policy}' model to {actor_timestep}.",
                        self.device_id,
                    )
                self.current_timestep[policy] = actor_timestep

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
                for policy, actor_model in self.actor_models.items():
                    filepath = os.path.join(
                        self.save_dir,
                        f"{policy}_actor_{self.current_timestep[policy]}.state_dict",
                    )
                    if self.verbose:
                        verbose_print(
                            f"Saving the '{policy}' (actor) torch model "
                            f"to the file: '{filepath}'.",
                            self.device_id,
                        )

                    torch.save(actor_model.state_dict(), filepath)

                for policy, critic_model in self.critic_models.items():
                    filepath = os.path.join(
                        self.save_dir,
                        f"{policy}_critic_{self.current_timestep[policy]}.state_dict",
                    )
                    if self.verbose:
                        verbose_print(
                            f"Saving the '{policy}' (critic) torch model "
                            f"to the file: '{filepath}'.",
                            self.device_id,
                        )

                    torch.save(critic_model.state_dict(), filepath)
