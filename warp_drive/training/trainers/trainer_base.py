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

from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.training.utils.ring_buffer import RingBufferManager
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


def verbose_print(message, device_id=None):
    if device_id is None:
        device_id = 0
    print(f"[Device {device_id}]: {message} ")


class TrainerBase:
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
        num_devices=1,
        device_id=0,
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
                a flag indicating whether there exist separate observations,
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
            num_devices: number of GPU devices used for (distributed) training.
                Defaults to 1.
            device_id: device ID. This is set in the context of multi-GPU training.
            results_dir: (optional) name of the directory to save results into.
            verbose:
                if False, training metrics are not printed to the screen.
                Defaults to True.
        """
        assert env_wrapper is not None
        assert not env_wrapper.env_backend == "cpu"
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
        # Sampler-related configurations (usually Optional)
        self.sample_params_schedules = {}
        sample_configs = self._get_config(["sampler", "params"]) if "sampler" in self.config else {}
        for param_name, param_config in sample_configs.items():
            self.sample_params_schedules[param_name] = ParamScheduler(param_config)

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

        # Number of GPU devices in the train
        self.num_devices = num_devices
        self.device_id = device_id

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
        assert self.num_episodes > 0
        self.training_batch_size = self._get_config(["trainer", "train_batch_size"])
        self.num_envs = self._get_config(["trainer", "num_envs"])

        self.neg_pos_env_ratio = self._get_config(["trainer", "neg_pos_env_ratio"]) if \
            "neg_pos_env_ratio" in self._get_config(["trainer"]) else -1

        self.training_batch_size_per_env = self.training_batch_size // self.num_envs
        assert self.training_batch_size_per_env > 0

        self.n_step = self.config["trainer"].get("n_step", 1)

        # Push all the data and tensor arrays to the GPU
        # upon resetting environments for the very first time.
        self.cuda_envs.reset_all_envs()

        if env_wrapper.env_backend == "pycuda":
            from warp_drive.managers.pycuda_managers.pycuda_function_manager import (
                PyCUDASampler,
            )

            self.cuda_sample_controller = PyCUDASampler(
                self.cuda_envs.cuda_function_manager
            )
        elif env_wrapper.env_backend == "numba":
            from warp_drive.managers.numba_managers.numba_function_manager import (
                NumbaSampler
            )

            self.cuda_sample_controller = NumbaSampler(
                self.cuda_envs.cuda_function_manager
            )

        # Create and push data placeholders to the device
        create_and_push_data_placeholders(
            env_wrapper=self.cuda_envs,
            action_sampler=self.cuda_sample_controller,
            policy_tag_to_agent_id_map=self.policy_tag_to_agent_id_map,
            create_separate_placeholders_for_each_policy=self.create_separate_placeholders_for_each_policy,
            obs_dim_corresponding_to_num_agents=self.obs_dim_corresponding_to_num_agents,
            training_batch_size_per_env=self.training_batch_size_per_env + self.n_step - 1,
        )
        # Seeding (device_id is included for distributed training)
        seed = (
            self.config["trainer"].get("seed", np.int32(time.time())) + self.device_id
        )
        self.cuda_sample_controller.init_random(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        self.cuda_envs.init_reset_pool(seed + random.randint(1, 10000))

        # For logging episodic reward
        self.num_completed_episodes = {}
        self.episodic_reward_sum = {}
        self.reward_running_sum = {}
        self.episodic_step_sum = {}
        self.step_running_sum = {}

        # Indicates the current timestep of the policy model
        self.current_timestep = {}

        self.total_steps = self.cuda_envs.episode_length * self.num_episodes
        self.num_iters = int(self.total_steps // self.training_batch_size)
        if self.num_iters == 0:
            raise ValueError(
                "Not enough steps to even perform a single training iteration!. "
                "Please increase the number of episodes or reduce the training "
                "batch size."
            )

        for policy in self.policies:
            self.current_timestep[policy] = 0
            self._initialize_policy_model(policy)

        # Load the model parameters (if model checkpoints are specified)
        # Note: Loading the model checkpoint may also update the current timestep!
        self.load_model_checkpoint()
        self.ddp_mode = {}
        for policy in self.policies:
            self._send_policy_model_to_device(policy)
            self._initialize_optimizer(policy)
            # Initialize episodic rewards and push to the GPU
            num_agents_for_policy = len(self.policy_tag_to_agent_id_map[policy])
            self.num_completed_episodes[policy] = 0
            self.episodic_reward_sum[policy] = (
                torch.tensor(0).type(torch.float32).cuda()
            )
            self.reward_running_sum[policy] = torch.zeros(
                (self.num_envs, num_agents_for_policy)
            ).cuda()
            self.episodic_step_sum[policy] = (
                torch.tensor(0).type(torch.int64).cuda()
            )
            self.step_running_sum[policy] = torch.zeros(self.num_envs, dtype=torch.int64).cuda()

        # Initialize the trainers
        self.trainers = {}
        self.clip_grad_norm = {}
        self.max_grad_norm = {}
        for policy in self.policies_to_train:
            self._initialize_policy_algorithm(policy)

        # Performance Stats
        self.perf_stats = PerfStats()

        # Metrics
        self.metrics = Metrics()
        self.sample_params_log = SamplerParamsLog()

        # Ring Buffer to save batch data
        self.ring_buffer = RingBufferManager()

    def _get_config(self, args):
        assert isinstance(args, (tuple, list))
        config = self.config
        for arg in args:
            try:
                config = config[arg]
            except ValueError:
                logging.error("Missing configuration '{arg}'!")
        return config

    # The followings are abstract classes that trainer class needs to finalize
    # They are mostly about how to manage and run the models
    def _initialize_policy_algorithm(self, policy):
        raise NotImplementedError

    def _initialize_policy_model(self, policy):
        raise NotImplementedError

    def _send_policy_model_to_device(self, policy):
        raise NotImplementedError

    def _initialize_optimizer(self, policy):
        raise NotImplementedError

    def _evaluate_policies(self, batch_index=0):
        raise NotImplementedError

    def _update_model_params(self, iteration):
        raise NotImplementedError

    def _load_model_checkpoint_helper(self, policy, ckpt_filepath):
        raise NotImplementedError

    def save_model_checkpoint(self, iteration=0):
        raise NotImplementedError

    # End of abstract classes

    def train(self):
        """
        Perform training.
        """
        # Ensure env is reset before the start of training, and done flags are False
        self.cuda_envs.reset_all_envs()

        for iteration in range(self.num_iters):
            start_time = time.time()

            # Generate a batched rollout for every CUDA environment.
            sample_params = self._generate_rollout_batch()

            # Train / update model parameters.
            metrics = self._update_model_params(iteration)

            self.perf_stats.iters = iteration + 1
            self.perf_stats.steps = self.perf_stats.iters * self.training_batch_size
            end_time = time.time()
            self.perf_stats.total_time += end_time - start_time

            # Log the training metrics
            self._log_metrics(metrics, sample_params)
            # Save torch model
            self.save_model_checkpoint(iteration)

    def _generate_rollout_batch(self):
        """
        Generate an environment rollout batch.
        """
        # Code timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        sample_params = self._get_sample_params()

        for batch_index in range(self.training_batch_size_per_env):

            # Evaluate policies to compute action probabilities
            start_event.record()
            probabilities = self._evaluate_policies(batch_index=batch_index)
            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.policy_eval_time += (
                start_event.elapsed_time(end_event) / 1000
            )

            # Sample actions using the computed probabilities
            # and push to the batch of actions
            start_event.record()
            self._sample_actions(probabilities, batch_index=batch_index, **sample_params)
            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.action_sample_time += (
                start_event.elapsed_time(end_event) / 1000
            )

            # Step through all the environments
            start_event.record()
            self.cuda_envs.step_all_envs()

            # Bookkeeping rewards and done flags
            _, done_flags = self._bookkeep_rewards_and_done_flags(batch_index=batch_index)

            # Reset all the environments that are in done state.
            if done_flags.any():
                self.cuda_envs.reset_only_done_envs()

            end_event.record()
            torch.cuda.synchronize()
            self.perf_stats.env_step_time += start_event.elapsed_time(end_event) / 1000

        return sample_params

    def _get_sample_params(self):
        sample_params = {}
        for param_name, param_schedule in self.sample_params_schedules.items():
            sample_params[param_name] = \
                param_schedule.get_param_value(self.current_timestep[self.policies[0]])
        return sample_params

    def _sample_actions(self, probabilities, batch_index=0, **sample_params):
        """
        Sample action probabilities (and push the sampled actions to the device).
        """
        assert isinstance(batch_index, int)
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                # Sample each individual policy
                policy_suffix = f"_{policy}"
                self._sample_actions_helper(
                    probabilities[policy], policy_suffix=policy_suffix, **sample_params
                )
                if batch_index >= 0:
                    # Push the actions to the batch
                    actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        _ACTIONS + policy_suffix
                    )
                    action_batch_name = f"{_ACTIONS}_batch" + policy_suffix
                    if self.ring_buffer.has(action_batch_name):
                        self.ring_buffer.get(action_batch_name).enqueue(actions)
                    else:
                        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                            name=action_batch_name
                        )[batch_index] = actions
        else:
            assert len(probabilities) == 1
            policy = list(probabilities.keys())[0]
            # sample a single or a combined policy
            self._sample_actions_helper(probabilities[policy], **sample_params)
            if batch_index >= 0:
                actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    _ACTIONS
                )
                # Push the actions to the batch, if action sampler has no policy tag
                # (1) there is only one policy, then action -> action_batch_policy
                # (2) there are multiple policies, then action[policy_tag_to_agent_id[policy]] -> action_batch_policy
                for policy in self.policies:
                    action_batch_name = f"{_ACTIONS}_batch_{policy}"
                    if len(self.policies) > 1:
                        agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                        if self.ring_buffer.has(action_batch_name):
                            self.ring_buffer.get(action_batch_name).enqueue(actions[:, agent_ids_for_policy])
                        else:
                            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                                name=action_batch_name
                            )[batch_index] = actions[:, agent_ids_for_policy]
                    else:
                        if self.ring_buffer.has(action_batch_name):
                            self.ring_buffer.get(action_batch_name).enqueue(actions)
                        else:
                            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                                name=action_batch_name
                            )[batch_index] = actions

    def _sample_actions_helper(self, probabilities, policy_suffix="", **sample_params):
        # Sample actions with policy_suffix tag
        num_action_types = len(probabilities)

        if num_action_types == 1:
            action_name = _ACTIONS + policy_suffix
            self.cuda_sample_controller.sample(
                self.cuda_envs.cuda_data_manager, probabilities[0], action_name, **sample_params
            )
        else:
            for action_type_id, probs in enumerate(probabilities):
                action_name = f"{_ACTIONS}_{action_type_id}" + policy_suffix
                self.cuda_sample_controller.sample(
                    self.cuda_envs.cuda_data_manager, probs, action_name, **sample_params
                )
                # Push (indexed) actions to 'actions'
                actions = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    action_name
                )
                self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    name=_ACTIONS + policy_suffix
                )[:, :, action_type_id] = actions[:, :, 0]

    def _bookkeep_rewards_and_done_flags(self, batch_index):
        """
        Push rewards and done flags to the corresponding batched versions.
        Also, update the episodic reward
        """
        assert isinstance(batch_index, int)
        # Push done flags to done_flags_batch
        # done_flags = (
        #     self.cuda_envs.cuda_data_manager.data_on_device_via_torch("_done_") > 0
        # )
        done_flags = self.cuda_envs.cuda_data_manager.data_on_device_via_torch("_done_")

        done_batch_name = f"{_DONE_FLAGS}_batch"
        if self.ring_buffer.has(done_batch_name):
            self.ring_buffer.get(done_batch_name).enqueue(done_flags)
        else:
            self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                name=done_batch_name
            )[batch_index] = done_flags

        done_env_ids = done_flags.nonzero()

        # Push rewards to rewards_batch and update the episodic rewards
        if self.create_separate_placeholders_for_each_policy:
            for policy in self.policies:
                rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                    f"{_REWARDS}_{policy}"
                )
                reward_batch_name = f"{_REWARDS}_batch_{policy}"
                if self.ring_buffer.has(reward_batch_name):
                    self.ring_buffer.get(reward_batch_name).enqueue(rewards)
                else:
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                        name=reward_batch_name
                    )[batch_index] = rewards

                # Update the episodic rewards
                self._update_episodic_rewards(rewards, done_env_ids, policy)

        else:
            rewards = self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                _REWARDS
            )
            for policy in self.policies:
                reward_batch_name = f"{_REWARDS}_batch_{policy}"
                if len(self.policies) > 1:
                    agent_ids_for_policy = self.policy_tag_to_agent_id_map[policy]
                    if self.ring_buffer.has(reward_batch_name):
                        self.ring_buffer.get(reward_batch_name).enqueue(rewards[:, agent_ids_for_policy])
                    else:
                        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                            name=reward_batch_name
                        )[batch_index] = rewards[:, agent_ids_for_policy]
                else:
                    if self.ring_buffer.has(reward_batch_name):
                        self.ring_buffer.get(reward_batch_name).enqueue(rewards)
                    else:
                        self.cuda_envs.cuda_data_manager.data_on_device_via_torch(
                            name=reward_batch_name
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
        self.step_running_sum[policy] += 1

        num_completed_episodes = len(done_env_ids)
        if num_completed_episodes > 0:
            # Update the episodic rewards
            self.episodic_reward_sum[policy] += torch.sum(
                self.reward_running_sum[policy][done_env_ids]
            )
            self.episodic_step_sum[policy] += torch.sum(
                self.step_running_sum[policy][done_env_ids]
            )
            self.num_completed_episodes[policy] += num_completed_episodes
            # Reset the reward running sum
            self.reward_running_sum[policy][done_env_ids] = 0
            self.step_running_sum[policy][done_env_ids] = 0

    def _log_metrics(self, metrics, sample_params=None):
        # Log the metrics if it is not empty
        if len(metrics) > 0:
            perf_stats = self.perf_stats.get_perf_stats()

            if self.verbose:
                print("\n")
                print("=" * 40)
                print(f"Device: {self.device_id}")
                print(
                    f"{'Iterations Completed':40}: "
                    f"{self.perf_stats.iters} / {self.num_iters}"
                )
                self.perf_stats.pretty_print(perf_stats)
                self.metrics.pretty_print(metrics)
                if sample_params is not None:
                    self.sample_params_log.pretty_print(sample_params)
                print("=" * 40, "\n")

            # Log metrics and performance stats
            logs = {"Iterations Completed": self.perf_stats.iters}
            logs.update(metrics)
            logs.update({"Perf. Stats": perf_stats})

            if self.num_devices > 1:
                fn = f"results_device_{self.device_id}.json"
            else:
                fn = "results.json"
            results_filename = os.path.join(self.save_dir, fn)
            if self.verbose:
                verbose_print(
                    f"Saving the results to the file '{results_filename}'",
                    self.device_id,
                )
            with open(results_filename, "a+", encoding="utf8") as fp:
                json.dump(logs, fp)
                fp.write("\n")
            self._heartbeat_check_printout(metrics)

    def _heartbeat_check_printout(self, metrics, check=False):
        if check is False:
            return

        if self.num_devices > 1:
            heartbeat_print = (
                "Iterations Completed: "
                + f"{self.perf_stats.iters} / {self.num_iters}: \n"
            )
            for policy in self.policies:
                heartbeat_print += (
                    f"policy '{policy}' - Mean episodic reward: "
                    f"{metrics[policy]['Mean episodic reward']} \n"
                )
            verbose_print(heartbeat_print, self.device_id)

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
                verbose_print(
                    "Loading the provided trainer model checkpoints.", self.device_id
                )
            for policy, ckpt_filepath in ckpts_dict.items():
                assert policy in self.policies
                self._load_model_checkpoint_helper(policy, ckpt_filepath)

    def graceful_close(self):
        # Delete the sample controller to clear
        # the random seeds defined in the CUDA memory heap.
        # Warning: Not closing gracefully could lead to a memory leak.
        del self.cuda_sample_controller
        if self.verbose:
            verbose_print("Trainer exits gracefully", self.device_id)

    def fetch_episode_states(
        self,
        list_of_states=None,  # list of states (data array names) to fetch
        env_id=0,  # environment id to fetch the states from
        include_rewards_actions=False,  # flag to output reward and action
        include_probabilities=False,  # flag to output action probability
        policy="",  # if include_rewards_actions=True, the corresponding policy tag if any
        **sample_params
    ):
        """
        Step through an env and fetch the desired states (data arrays on the GPU)
        for an entire episode. The trained models will be used for evaluation.
        """
        assert 0 <= env_id < self.num_envs
        if list_of_states is None:
            list_of_states = []
        assert isinstance(list_of_states, list)

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

        if include_rewards_actions:
            policy_suffix = f"_{policy}" if len(policy) > 0 else ""
            action_name = _ACTIONS + policy_suffix
            reward_name = _REWARDS + policy_suffix
            # Note the size is 1 step smaller than states because we do not have r_0 and a_T
            episode_actions = np.zeros(
                (
                    env.episode_length, *self.cuda_envs.cuda_data_manager.get_shape(action_name)[1:]
                ),
                dtype=self.cuda_envs.cuda_data_manager.get_dtype(action_name)
            )
            episode_rewards = np.zeros(
                (
                    env.episode_length, *self.cuda_envs.cuda_data_manager.get_shape(reward_name)[1:]
                ),
                dtype=np.float32)

        if include_probabilities:
            episode_probabilities = {}

        for timestep in range(env.episode_length):
            # Update the episode states s_t
            for state in list_of_states:
                episode_states[state][
                    timestep
                ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                    env_id
                ]
            # Evaluate policies to compute action probabilities, we set batch_index=-1 to avoid batch writing
            probabilities = self._evaluate_policies(batch_index=-1)

            # Sample actions, we set batch_index=-1 to avoid batch writing
            self._sample_actions(probabilities, batch_index=-1, **sample_params)

            # Step through all the environments
            self.cuda_envs.step_all_envs()

            if include_rewards_actions:
                # Update the episode action a_t
                episode_actions[timestep] = \
                    self.cuda_envs.cuda_data_manager.pull_data_from_device(action_name)[env_id]
                # Update the episode reward r_(t+1)
                episode_rewards[timestep] = \
                    self.cuda_envs.cuda_data_manager.pull_data_from_device(reward_name)[env_id]
            if include_probabilities:
                # Update the episode action probability p_t
                if len(policy) > 0:
                    probs = {policy: [p[env_id].detach().cpu().numpy() for p in probabilities[policy]]}
                else:
                    probs = {}
                    for policy, value in probabilities.items():
                        probs[policy] = [v[env_id].detach().cpu().numpy() for v in value]
                episode_probabilities[timestep] = probs
            # Fetch the states when episode is complete
            if env.cuda_data_manager.pull_data_from_device("_done_")[env_id]:
                for state in list_of_states:
                    episode_states[state][
                        timestep + 1
                    ] = self.cuda_envs.cuda_data_manager.pull_data_from_device(state)[
                        env_id
                    ]
                break
        if include_rewards_actions and not include_probabilities:
            return episode_states, episode_actions, episode_rewards
        elif include_rewards_actions and include_probabilities:
            return episode_states, episode_actions, episode_rewards, episode_probabilities
        else:
            return episode_states

    def evaluate_episodes(
        self,
        **sample_params,
    ):
        """
        Step through all envs by one episode and fetch the rewards for evaluation.
        """
        episodic_reward_sum = {}
        episodic_step_sum = {}
        for policy in self.policies:
            num_agents_for_policy = len(self.policy_tag_to_agent_id_map[policy])
            episodic_reward_sum[policy] = np.zeros((self.num_envs, num_agents_for_policy), dtype=np.float32)
            episodic_step_sum[policy] = np.zeros(self.num_envs, dtype=np.int32)

        self.cuda_envs.reset_all_envs()

        for _ in range(self.cuda_envs.episode_length):
            # Evaluate policies to compute action probabilities, we set batch_index=-1 to avoid batch writing
            probabilities = self._evaluate_policies(batch_index=-1)

            # Sample actions, we set batch_index=-1 to avoid batch writing
            self._sample_actions(probabilities, batch_index=-1, **sample_params)

            # Step through all the environments
            self.cuda_envs.step_all_envs()

            undone_flags = (
                    self.cuda_envs.cuda_data_manager.data_on_device_via_torch("_done_") == 0
            ).cpu().numpy()
            if not undone_flags.any():
                break

            if self.create_separate_placeholders_for_each_policy:
                for policy in self.policies:
                    rewards = self.cuda_envs.cuda_data_manager.pull_data_from_device(
                        f"{_REWARDS}_{policy}"
                    )
                    episodic_reward_sum[policy][undone_flags] += rewards[undone_flags]
                    episodic_step_sum[policy][undone_flags] += 1
            else:
                rewards = self.cuda_envs.cuda_data_manager.pull_data_from_device(
                    _REWARDS
                )
                for policy in self.policies:
                    episodic_reward_sum[policy][undone_flags] += \
                        rewards[undone_flags][self.policy_tag_to_agent_id_map[policy]]
                    episodic_step_sum[policy][undone_flags] += 1

            # Reset all the environments that are in done state,
            # do not undo done anymore because we just need one episode
            self.cuda_envs.reset_only_done_envs(undo_done_after_reset=False)

        return episodic_reward_sum, episodic_step_sum


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

    @staticmethod
    def pretty_print(stats):
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


class SamplerParamsLog:

    def __init__(self):
        pass

    def pretty_print(self, params):
        assert params is not None
        assert isinstance(params, dict)
        if len(params) > 0:
            print("=" * 40)
            print(f"Parameters for sampler")
            print("=" * 40)
            for k, v in params.items():
                print(f"{k:40}: {v:10.5f}")
