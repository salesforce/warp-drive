# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The Pytorch Lightning-based Trainer, PerfStats and Metrics classes
Pytorch Lightning: https://www.pytorchlightning.ai/

Here, we integrate the WarpDrive trainer with the Pytorch Lightning framework,
which greatly reduces the trainer boilerplate code, and improves training flexibility.
"""

import argparse
import json
import logging
import os
import time
from typing import Callable, Iterable, Tuple

import numpy as np
import torch
import yaml
from gym.spaces import Discrete, MultiDiscrete
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from warp_drive.managers.function_manager import CUDASampler
from warp_drive.training.algorithms.a2c import A2C
from warp_drive.training.algorithms.ppo import PPO
from warp_drive.training.models.fully_connected import FullyConnected
from warp_drive.training.trainer import Metrics
from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.training.utils.param_scheduler import LRScheduler, ParamScheduler
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


def verbose_print(message, device_id=None):
    if device_id is None:
        device_id = 0
    print(f"[Device {device_id}]: {message} ")


class WarpDriveDataset(Dataset):
    """
    The WarpDrive dataset class.
    """

    def __init__(self, generate_data: Callable, batch_size):
        self.generate_data = generate_data
        self.batch_size = batch_size
        self.data_dict = None  # this will be set later (see below)

    def __getitem__(self, batch_index=0) -> Iterable:
        self.data_dict = self.generate_data(batch_index=batch_index)
        return self.data_dict

    def __len__(self):
        return self.batch_size


class WarpDriveModule(LightningModule):
    """
    The trainer object using Pytorch Lightning training APIs.
    Pytorch Lightning: https://www.pytorchlightning.ai/
    """

    def __init__(
        self,
        env_wrapper=None,
        config=None,
        policy_tag_to_agent_id_map=None,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
        results_dir=None,
        verbose=True,
    ):
        """
        Args:
            env_wrapper: the wrapped environment object.
            config: the experiment run configuration.
            policy_tag_to_agent_id_map:
                a dictionary mapping policy tag to agent ids.
            create_separate_placeholders_for_each_policy:
                flag indicating whether there exist separate observations,
                actions and rewards placeholders, for each policy,
                as designed in the step function. The placeholders will be
                used in the step() function and during training.
                When there's only a single policy, this flag will be False.
                It can also be True when there are multiple policies, yet
                all the agents have the same obs and action space shapes,
                so we can share the same placeholder.
                Defaults to "False".
            obs_dim_corresponding_to_num_agents:
                indicative of which dimension in the observation corresponds
                to the number of agents, as designed in the step function.
                It may be "first" or "last". In other words,
                observations may be shaped (num_agents, *feature_dim) or
                (*feature_dim, num_agents). This is required in order for
                WarpDrive to process the observations correctly. This is only
                relevant when a single obs key corresponds to multiple agents.
                Defaults to "first".
            results_dir: (optional) name of the directory to save results into.
            verbose:
                if enabled, training metrics are printed to the screen.

        """
        super().__init__()

        assert env_wrapper is not None
        assert env_wrapper.use_cuda
        assert config is not None
        assert isinstance(create_separate_placeholders_for_each_policy, bool)
        assert obs_dim_corresponding_to_num_agents in ["first", "last"]
        self.obs_dim_corresponding_to_num_agents = obs_dim_corresponding_to_num_agents

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

        if results_dir is None:
            # Use the current time as the name for the results directory.
            results_dir = f"{time.time():10.0f}"

        # Directory to save model checkpoints and metrics
        self.save_dir = os.path.join(
            self._get_config(["saving", "basedir"]),
            self._get_config(["saving", "name"]),
            self._get_config(["saving", "tag"]),
            results_dir,
        )
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # Save the run configuration
        config_filename = os.path.join(self.save_dir, "run_config.json")
        with open(config_filename, "a+", encoding="utf8") as fp:
            json.dump(self.config, fp)
            fp.write("\n")

        # Flag to determine whether to print training metrics
        self.verbose = verbose

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
        self.iters = 0  # iteration counter
        self.num_episodes = self._get_config(["trainer", "num_episodes"])
        assert self.num_episodes > 0
        self.training_batch_size = self._get_config(["trainer", "train_batch_size"])
        self.num_envs = self._get_config(["trainer", "num_envs"])

        self.training_batch_size_per_env = self.training_batch_size // self.num_envs
        assert self.training_batch_size_per_env > 0

        # Push all the data and tensor arrays to the GPU
        # upon resetting environments for the very first time.
        self.cuda_envs.reset_all_envs()

        # Create and push data placeholders to the device
        create_and_push_data_placeholders(
            self.cuda_envs,
            self.policy_tag_to_agent_id_map,
            self.create_separate_placeholders_for_each_policy,
            self.obs_dim_corresponding_to_num_agents,
            self.training_batch_size_per_env,
            push_data_batch_placeholders=False,
        )

        self.cuda_sample_controller = CUDASampler(self.cuda_envs.cuda_function_manager)

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

        # Seeding
        seed = self.config["trainer"].get("seed", np.int32(time.time()))
        seed_everything(seed)
        self.cuda_sample_controller.init_random(seed)

        # Define models
        self.models = {}

        # For logging episodic reward
        self.num_completed_episodes = {}
        self.episodic_reward_sum = {}
        self.reward_running_sum = {}

        # Indicates the current timestep of the policy model
        self.current_timestep = {}

        self.total_steps = self.cuda_envs.episode_length * self.num_episodes
        self.num_iters = int(self.total_steps // self.training_batch_size)

        for policy in self.policies:
            self.current_timestep[policy] = 0
            self._initialize_policy_model(policy)

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
            self._initialize_policy_algorithm(policy)

        # Metrics
        self.metrics = Metrics()

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
        if policy_model_config["type"] == "fully_connected":
            model = FullyConnected(
                self.cuda_envs,
                policy_model_config["fc_dims"],
                policy,
                self.policy_tag_to_agent_id_map,
                self.create_separate_placeholders_for_each_policy,
                self.obs_dim_corresponding_to_num_agents,
            )
        else:
            raise NotImplementedError
        self.models[policy] = model

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
                "Only 'Discrete' or 'MultiDiscrete' type action spaces are supported!"
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

    def _sample_actions(self, probabilities):
        """
        Sample action probabilities (and push the sampled actions to the device).
        """
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                suffix = f"_{policy}"
                self._sample_actions_helper(probabilities[policy], suffix=suffix)
        else:
            assert len(probabilities) == 1
            policy = list(probabilities.keys())[0]
            self._sample_actions_helper(probabilities[policy])

    def _sample_actions_helper(self, probabilities, suffix=""):

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
                name=f"{_ACTIONS}" + suffix
            )[:, :, num_action_types - 1] = actions

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
                    name=f"{_ACTIONS}" + suffix
                )[:, :, action_idx] = actions

    def _bookkeep_rewards_and_done_flags(self):
        """
        Push rewards and done flags to the corresponding batched versions.
        Also, update the episodic reward
        """
        # Push done flags to done_flags_batch
        done_flags = (
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch("_done_") > 0
        )

        done_env_ids = done_flags.nonzero()

        # Push rewards to rewards_batch and update the episodic rewards
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_{policy}"
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=f"{_REWARDS}_{policy}"
                )[:] = rewards

                # Update the episodic rewards
                self._update_episodic_rewards(rewards, done_env_ids, policy)

        else:
            rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                _REWARDS
            )

            # Update the episodic rewards
            # (sum of individual step rewards over an episode)
            for policy in self.policies:
                self._update_episodic_rewards(
                    rewards[:, self.policy_tag_to_agent_id_map[policy]],
                    done_env_ids,
                    policy,
                )
        return done_flags

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

            if self.verbose:
                self.metrics.pretty_print(metrics)

            results_filename = os.path.join(self.save_dir, "results.json")
            if self.verbose:
                verbose_print(f"Saving the results to the file '{results_filename}'")
            with open(results_filename, "a+", encoding="utf8") as fp:
                json.dump(metrics, fp)
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
            if self.verbose:
                verbose_print("Loading the provided trainer model checkpoints.")
            for policy, ckpt_filepath in ckpts_dict.items():
                assert policy in self.policies
                self._load_model_checkpoint_helper(policy, ckpt_filepath)

    def _load_model_checkpoint_helper(self, policy, ckpt_filepath):
        if ckpt_filepath != "":
            assert os.path.isfile(ckpt_filepath), "Invalid model checkpoint path!"
            if self.verbose:
                verbose_print(
                    f"Loading the '{policy}' torch model "
                    f"from the previously saved checkpoint: '{ckpt_filepath}'"
                )
            self.models[policy].load_state_dict(torch.load(ckpt_filepath))

            # Update the current timestep using the saved checkpoint filename
            timestep = int(ckpt_filepath.split(".state_dict")[0].split("_")[-1])
            if self.verbose:
                verbose_print(
                    f"Updating the timestep for the '{policy}' model to {timestep}.",
                )
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
                if self.verbose:
                    verbose_print(
                        f"Saving the '{policy}' torch model "
                        f"to the file: '{filepath}'."
                    )

                torch.save(model.state_dict(), filepath)

    def graceful_close(self):
        # Delete the sample controller to clear
        # the random seeds defined in the CUDA memory heap.
        # Warning: Not closing gracefully could lead to a memory leak.

        del self.cuda_sample_controller
        if self.verbose:
            verbose_print("Trainer exits gracefully")

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

        for timestep in range(env.episode_length):
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
                        timestep + 1
                    ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                        env_id
                    ]
                break

        return episode_states

    def _generate_rollout(self, start_event, end_event, batch_index=0):
        """
        Perform a forward pass through the model(s) and step through the environment.
        """

        # Evaluate policies to compute action probabilities
        start_event.record()
        probabilities = self._evaluate_policies(batch_index=batch_index)
        end_event.record()
        torch.cuda.synchronize()

        # Sample actions using the computed probabilities
        # and push to the batch of actions
        start_event.record()
        self._sample_actions(probabilities)
        end_event.record()
        torch.cuda.synchronize()

        # Step through all the environments
        start_event.record()
        self.cuda_envs.step_all_envs()

        # Bookkeeping rewards and done flags
        done_flags = self._bookkeep_rewards_and_done_flags()

        # Reset all the environments that are in done state.
        if done_flags.any():
            self.cuda_envs.reset_only_done_envs()

        end_event.record()
        torch.cuda.synchronize()

    def _generate_training_data(self, batch_index=0):
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

        # Evaluate policies and run step functions
        self._generate_rollout(start_event, end_event, batch_index=batch_index)

        # Fetch the actions and rewards batches for all agents
        if not self.create_separate_placeholders_for_each_policy:
            all_actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                f"{_ACTIONS}"
            )
            all_rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                f"{_REWARDS}"
            )
        done_flags = self.cuda_envs.cuda_data_manager.data_on_device_via_torch("_done_")
        # On the device, observations_batch, actions_batch,
        # rewards_batch are all shaped
        # (batch_size, num_envs, num_agents, *feature_dim).
        # done_flags_batch is shaped (batch_size, num_envs)
        # Perform training sequentially for each policy
        training_batch = {}
        for policy in self.policies_to_train:
            if self.create_separate_placeholders_for_each_policy:
                actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_ACTIONS}{policy}"
                )
                rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_{policy}"
                )
            else:
                # Filter the actions and rewards only for the agents
                # corresponding to a particular policy
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                actions = all_actions[:, agent_ids_for_policy, :]
                rewards = all_rewards[:, agent_ids_for_policy]

            # Fetch the (processed) observations batch to pass through the model
            processed_obs = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                f"{_PROCESSED_OBSERVATIONS}_batch_{policy}"
            )

            training_batch[policy] = (
                actions,
                rewards,
                done_flags,
                processed_obs[batch_index],
            )
        return training_batch

    # APIs to integrate with Pytorch Lightning
    # ----------------------------------------
    def train_dataloader(self):
        """Get the train data loader"""
        dataset = WarpDriveDataset(
            self._generate_training_data, batch_size=self.training_batch_size_per_env
        )
        return DataLoader(dataset, batch_size=self.training_batch_size_per_env)

    def configure_optimizers(self):
        """Optimizers and LR Schedules"""
        optimizers = []
        lr_schedules = []

        for policy in self.policies_to_train:
            # Initialize the (ADAM) optimizer
            lr_schedule = self._get_config(["policy", policy, "lr"])
            init_timestep = self.current_timestep[policy]
            initial_lr = ParamScheduler(lr_schedule).get_param_value(init_timestep)
            optimizer = torch.optim.Adam(
                self.models[policy].parameters(), lr=initial_lr
            )

            # Initialize the learning rate scheduler
            lr_scheduler = LRScheduler(
                lr_schedule,
                optimizer,
                init_timestep,
                timesteps_per_iteration=self.training_batch_size,
            )

            lr_schedules += [{"scheduler": lr_scheduler}]
            optimizers += [optimizer]
        return optimizers, lr_schedules

    def configure_gradient_clipping(
        self,
        optimizer,
        optimizer_idx,
        gradient_clip_val=None,
        gradient_clip_algorithm=None,
    ):
        policy = self.policies_to_train[optimizer_idx]
        if gradient_clip_val is None:
            gradient_clip_val = self.max_grad_norm[policy]
        if self.clip_grad_norm[policy]:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=gradient_clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm,
            )

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx=0, optimizer_idx=0
    ):
        """
        Carries out a single training step based on a batch of rollout data.
        Args:
            batch of sampled actions, rewards, done flags and processed observations.
        Returns: loss.
        """
        assert batch_idx >= 0
        assert optimizer_idx >= 0

        if optimizer_idx == 0:
            # Do this only once for all the optimizers
            self.iters += 1

        # Flag for logging (which also happens after the last iteration)
        logging_flag = (
            self.iters % self.config["saving"]["metrics_log_freq"] == 0
            or self.iters == self.num_iters - 1
        )

        policy = self.policies_to_train[optimizer_idx]

        actions_batch, rewards_batch, done_flags_batch, processed_obs_batch = batch[
            policy
        ]

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

        # Logging
        if logging_flag:
            assert isinstance(metrics, dict)
            # Update the metrics dictionary
            metrics.update(
                {
                    "Current timestep": self.current_timestep[policy],
                    "Gradient norm": grad_norm,
                    "Mean episodic reward": self.episodic_reward_sum[policy].item()
                    / (self.num_completed_episodes[policy] + _EPSILON),
                }
            )

            # Reset sum and counter
            self.episodic_reward_sum[policy] = (
                torch.tensor(0).type(torch.float32).cuda()
            )
            self.num_completed_episodes[policy] = 0

            self._log_metrics({policy: metrics})

            # Logging
            self.log(
                f"loss_{policy}", loss, prog_bar=True, on_step=False, on_epoch=True
            )
            for key in metrics:
                self.log(
                    f"{key}_{policy}",
                    metrics[key],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )

        # Save the model checkpoint
        self.save_model_checkpoint(self.iters)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no cover
        parser = argparse.ArgumentParser(parents=[parent_parser])
        return parser


class PerfStatsCallback(Callback):
    """
    Performance stats that will be included in rollout metrics.
    """

    def __init__(self, batch_size=None, num_iters=None, log_freq=1):
        assert batch_size is not None
        assert num_iters is not None
        super().__init__()
        self.batch_size = batch_size
        self.num_iters = num_iters
        self.log_freq = log_freq
        self.iters = 0
        self.steps = 0
        self.training_time = 0.0

        # For timing purposes
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)

    def get_perf_stats(self):
        return {
            "Mean training time per iter (ms)": self.training_time * 1000 / self.iters,
            "Mean steps per sec (training time)": self.steps / self.training_time,
        }

    def pretty_print(self, stats):
        print("=" * 40)
        print("Speed performance stats")
        print("=" * 40)
        iteration_str = "Iteration"
        iter_counter = f"{self.iters} / {self.num_iters}"
        print(f"{iteration_str:40}: {iter_counter:13}")
        for k, v in stats.items():
            print(f"{k:40}: {v:10.2f}")
        print("\n")

    # Pytorch Lightning hooks
    def on_batch_start(self, trainer=None, pl_module=None):
        assert trainer is not None
        assert pl_module is not None
        self.iters += 1
        self.steps = self.iters * self.batch_size
        self.start_event.record()

    def on_batch_end(self, trainer=None, pl_module=None):
        assert trainer is not None
        assert pl_module is not None
        self.end_event.record()
        torch.cuda.synchronize()

        self.training_time += self.start_event.elapsed_time(self.end_event) / 1000

        if self.iters % self.log_freq == 0 or self.iters == self.num_iters:
            self.pretty_print(self.get_perf_stats())


class CudaCallback(Callback):
    """
    Callbacks pertaining to CUDA.
    """

    def __init__(self, module):
        self.module = module

    # Pytorch Lightning hooks
    def on_train_start(self, trainer=None, pl_module=None):
        assert trainer is not None
        assert pl_module is not None
        self.module.cuda_envs.reset_all_envs()

    def on_train_end(self, trainer=None, pl_module=None):
        assert trainer is not None
        assert pl_module is not None
        print("Training is complete!")
