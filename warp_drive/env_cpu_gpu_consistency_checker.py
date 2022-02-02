# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Consistency tests for comparing the cuda (gpu) / no cuda (cpu) version
"""

import logging

import numpy as np
import torch
from gym.spaces import Discrete, MultiDiscrete
from gym.utils import seeding

from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.utils.data_loader import create_and_push_data_placeholders
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS


def generate_random_actions(env, num_envs, seed=None):
    """
    Generate random actions for each agent and each env.
    """
    agent_ids = sorted(list(env.action_space.keys()))
    if seed is not None:
        np_random = seeding.np_random(seed)[0]
    else:
        np_random = np.random

    return [
        {
            agent_id: _generate_random_actions_helper(
                env.action_space[agent_id], np_random
            )
            for agent_id in agent_ids
        }
        for _ in range(num_envs)
    ]


def _generate_random_actions_helper(action_space, np_random):
    if isinstance(action_space, Discrete):
        return np_random.randint(low=0, high=int(action_space.n), dtype=np.int32)
    if isinstance(action_space, MultiDiscrete):
        return np_random.randint(
            low=[0] * len(action_space.nvec),
            high=action_space.nvec,
            dtype=np.int32,
        )
    raise NotImplementedError(
        "Only 'Discrete' or 'MultiDiscrete' type action spaces are supported"
    )


class EnvironmentCPUvsGPU:
    """
    test the rollout consistency between the CPU environment and the GPU environment
    """

    def __init__(
        self,
        cpu_env_class=None,
        cuda_env_class=None,
        dual_mode_env_class=None,
        env_configs=None,
        num_envs=3,
        num_episodes=2,
        use_gpu_testing_mode=True,
        env_registry=None,
        env_wrapper=EnvWrapper,
        policy_tag_to_agent_id_map=None,
        create_separate_placeholders_for_each_policy=False,
    ):
        """
        :param cpu_env_class: cpu env class to test, for example, TagGridWorld
        :param cuda_env_class: cuda env class to test, for example, CUDATagGridWorld
        :param dual_mode_env_class: env class that supports both cpu and cuda
        :param env_config: env configuration
        :param num_envs: number of parallel example_envs in the test.
            If use_gpu_testing_mode = True,
            num_envs = 2 and num_agents=5 are enforced
        :param num_episodes: number of episodes in the test
            hint: number >=2 is recommended
            since it can fully test the reset
        :param use_gpu_testing_mode: determine whether to simply load the
            discrete_and_continuous_tag_envs.cubin or compile the .cu source
            code to create a .cubin.
            If use_gpu_testing_mode = True: do not forget to
            include your testing env into discrete_and_continuous_tag_envs.cu
            and build it. This is the recommended flow because the
            Makefile will automate this build.
        :param env_registry: EnvironmentRegistrar object
            it provides the customized env info (like src path) for the build
        :param policy_tag_to_agent_id_map:
            a dictionary mapping policy tag to agent ids
        :param create_separate_placeholders_for_each_policy:
            flag indicating whether there exist separate observations,
            actions and rewards placeholders, for each policy,
            as designed in the step function. The placeholders will be
            used in the step() function and during training.
            When there's only a single policy, this flag will be False.
            It can also be True when there are multiple policies, yet
            all the agents have the same obs and action space shapes,
            so we can share the same placeholder.
            Defaults to "False"

        """
        if dual_mode_env_class is not None:
            self.cpu_env_class = dual_mode_env_class
            self.cuda_env_class = dual_mode_env_class
        else:
            assert cpu_env_class is not None and cuda_env_class is not None
            self.cpu_env_class = cpu_env_class
            self.cuda_env_class = cuda_env_class

        self.env_configs = env_configs
        if use_gpu_testing_mode:
            logging.warning(
                f"enforce num_envs = {num_envs} because you have "
                f"use_gpu_testing_mode = True, where the cubin file"
                f"supporting this testing mode assumes 2 parallel example_envs"
            )
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.use_gpu_testing_mode = use_gpu_testing_mode
        self.env_registry = env_registry
        self.env_wrapper = env_wrapper
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.create_separate_placeholders_for_each_policy = (
            create_separate_placeholders_for_each_policy
        )

    def test_env_reset_and_step(self, consistency_threshold_pct=1, seed=None):
        """
        Perform consistency checks for the reset() and step() functions
        consistency_threshold_pct: consistency threshold as a percentage
        (defaults to 1%).
        """
        for scenario in self.env_configs:
            env_config = self.env_configs[scenario]

            print(f"Running {scenario}...")

            # Env Reset
            # CPU version of env
            # ------------------
            env_cpu = {}
            obs_cpu = []
            for env_id in range(self.num_envs):
                if self.env_registry is not None and self.env_registry.has_env(
                    self.cpu_env_class.name, device="cpu"
                ):
                    env_cpu[env_id] = self.env_registry.get(
                        self.cpu_env_class.name, use_cuda=False
                    )(**env_config)
                else:
                    env_cpu[env_id] = self.cpu_env_class(**env_config)
                if self.use_gpu_testing_mode:
                    assert env_cpu[env_id].num_agents == 5

                # obs will be a dict of {agent_id: agent_specific_obs}
                # agent_specific_obs could itself be a dictionary or an array
                obs_cpu += [env_cpu[env_id].reset()]

            # GPU version of env
            # ------------------
            if self.env_registry is not None and self.env_registry.has_env(
                self.cuda_env_class.name, device="cuda"
            ):
                # we use env_registry to instantiate env object
                env_obj = None
            else:
                # if not registered, we use the brutal force
                env_obj = self.cuda_env_class(**env_config)
            env_gpu = self.env_wrapper(
                env_obj=env_obj,
                env_name=self.cuda_env_class.name,
                env_config=env_config,
                num_envs=self.num_envs,
                use_cuda=True,
                testing_mode=self.use_gpu_testing_mode,
                env_registry=self.env_registry,
            )
            env_gpu.reset_all_envs()

            # Push obs, sampled actions, rewards and done flags placeholders
            # to the device
            create_and_push_data_placeholders(
                env_gpu,
                self.policy_tag_to_agent_id_map,
                self.create_separate_placeholders_for_each_policy,
            )

            # if the environment has explicit definition about
            # how action index to the action step
            if hasattr(env_gpu.env, "step_actions"):
                k_index_to_action_arr = env_gpu.env.step_actions
                env_gpu.cuda_data_manager.add_shared_constants(
                    {"kIndexToActionArr": k_index_to_action_arr}
                )
                env_gpu.cuda_function_manager.initialize_shared_constants(
                    env_gpu.cuda_data_manager, constant_names=["kIndexToActionArr"]
                )

            # Consistency checks after the first reset
            # ----------------------------------------
            logging.info("Running obs consistency check after the first reset...")
            self._run_obs_consistency_checks(
                obs_cpu, env_gpu, threshold_pct=consistency_threshold_pct
            )

            # Consistency checks during subsequent steps and resets
            # -----------------------------------------------------
            logging.info(
                "Running obs/rew/done consistency check "
                "during subsequent env steps and resets..."
            )

            # Test across multiple episodes
            for _ in range(self.num_episodes * env_gpu.episode_length):
                actions_list_of_dicts = generate_random_actions(
                    env_cpu[env_id], self.num_envs, seed
                )

                # Push the actions to the device (GPU)
                if self.create_separate_placeholders_for_each_policy:
                    for policy in self.policy_tag_to_agent_id_map:
                        suffix = f"_{policy}"
                        self._push_actions_to_device(
                            actions_list_of_dicts, env_gpu, suffix
                        )
                else:
                    self._push_actions_to_device(actions_list_of_dicts, env_gpu)

                obs_list = []
                rew_list = []
                done_list = []

                for env_id in range(self.num_envs):
                    obs, rew, done, _ = env_cpu[env_id].step(
                        actions_list_of_dicts[env_id]
                    )

                    obs_list += [obs]

                    combined_rew_array = np.stack(
                        [rew[key] for key in sorted(rew.keys())], axis=0
                    )
                    rew_list += [combined_rew_array]
                    done_list += [done]

                rew_cpu = np.stack(rew_list, axis=0)

                done_cpu = {
                    "__all__": np.array([done["__all__"] for done in done_list])
                }

                # Step through all the environments
                env_gpu.step_all_envs()

                rew_gpu = env_gpu.cuda_data_manager.pull_data_from_device(_REWARDS)
                done_gpu = (
                    env_gpu.cuda_data_manager.data_on_device_via_torch("_done_")
                    .cpu()
                    .numpy()
                )

                self._run_obs_consistency_checks(
                    obs_list, env_gpu, threshold_pct=consistency_threshold_pct
                )
                self._run_consistency_checks(
                    rew_cpu, rew_gpu, threshold_pct=consistency_threshold_pct
                )
                assert all(done_cpu["__all__"] == (done_gpu > 0))

                # GPU reset
                env_gpu.reset_only_done_envs()

                # Now, pull done flags, and assert that they are set to 0 (False) again
                done_gpu = env_gpu.cuda_data_manager.pull_data_from_device("_done_")
                assert done_gpu.sum() == 0

                # CPU reset
                env_reset = False  # flag to indicate if any env is reset
                for env_id in range(self.num_envs):
                    if done_cpu["__all__"][env_id]:
                        env_reset = True
                        # Reset the CPU for this env_id
                        obs_list[env_id] = env_cpu[env_id].reset()

                if env_reset:
                    # Run obs consistency checks when any env was reset
                    self._run_obs_consistency_checks(
                        obs_list, env_gpu, threshold_pct=consistency_threshold_pct
                    )

            print(
                f"The CPU and the GPU environment outputs are consistent "
                f"within {consistency_threshold_pct} percent."
            )

    def _push_actions_to_device(self, actions_list_of_dicts, env_gpu, suffix=""):
        actions_list = []
        for action_dict in actions_list_of_dicts:
            if suffix == "":
                agent_ids = list(range(env_gpu.n_agents))
            else:
                policy = suffix.split("_")[1]
                agent_ids = sorted(self.policy_tag_to_agent_id_map[policy])

            combined_actions = np.stack(
                [action_dict[agent_id] for agent_id in agent_ids], axis=0
            )
            actions_list += [combined_actions]
        actions = np.stack(actions_list, axis=0)
        name = _ACTIONS + suffix
        actions_data = DataFeed()
        actions_data.add_data(name=name, data=actions)
        assert env_gpu.cuda_data_manager.is_data_on_device_via_torch(name)
        env_gpu.cuda_data_manager.data_on_device_via_torch(name)[:] = torch.from_numpy(
            actions
        )

    def _get_cpu_gpu_obs(self, obs, env_gpu, policy_tag_to_agent_id_map, suffix=""):
        if suffix == "":
            agent_ids = list(range(env_gpu.n_agents))
            first_agent_id = agent_ids[0]
        else:
            pol_tag = suffix.split("_")[1]
            agent_ids = sorted(policy_tag_to_agent_id_map[pol_tag])
            first_agent_id = agent_ids[0]

        num_envs = env_gpu.n_envs
        first_env_id = 0

        if isinstance(obs[first_env_id][first_agent_id], (list, np.ndarray)):
            obs_cpu = {
                "obs": np.stack(
                    [
                        np.array([obs[env_id][agent_id] for agent_id in agent_ids])
                        for env_id in range(num_envs)
                    ],
                    axis=0,
                )
            }
            obs_gpu = {
                "obs": env_gpu.cuda_data_manager.pull_data_from_device(
                    f"{_OBSERVATIONS}" + suffix
                )
            }
        elif isinstance(obs[first_env_id][first_agent_id], dict):
            obs_cpu = {}
            obs_gpu = {}
            for key in obs[first_env_id][first_agent_id]:
                obs_cpu[key] = np.stack(
                    [
                        np.array([obs[env_id][agent_id][key] for agent_id in agent_ids])
                        for env_id in range(num_envs)
                    ],
                    axis=0,
                )
                obs_gpu[key] = env_gpu.cuda_data_manager.pull_data_from_device(
                    f"{_OBSERVATIONS}" + suffix + f"_{key}"
                )
        else:
            raise ValueError("obs may be an array-type or a dictionary")

        assert isinstance(obs_cpu, dict)
        assert isinstance(obs_gpu, dict)
        assert sorted(list(obs_cpu.keys())) == sorted(list(obs_gpu.keys()))
        return obs_cpu, obs_gpu

    def _run_obs_consistency_checks(self, obs, env_gpu, threshold_pct):
        if self.create_separate_placeholders_for_each_policy:
            assert len(self.policy_tag_to_agent_id_map) > 1
            for pol_mod_tag in self.policy_tag_to_agent_id_map:
                obs_cpu, obs_gpu = self._get_cpu_gpu_obs(
                    obs,
                    env_gpu,
                    self.policy_tag_to_agent_id_map,
                    suffix=f"_{pol_mod_tag}",
                )
                for key, val in obs_cpu.items():
                    self._run_consistency_checks(val, obs_gpu[key], threshold_pct)
        else:
            obs_cpu, obs_gpu = self._get_cpu_gpu_obs(
                obs, env_gpu, self.policy_tag_to_agent_id_map
            )
            for key, val in obs_cpu.items():
                self._run_consistency_checks(val, obs_gpu[key], threshold_pct)

    @staticmethod
    def _run_consistency_checks(cpu_value, gpu_value, threshold_pct=1):
        """
        Perform consistency checks between the cpu and gpu values.
        The default threshold is 2 decimal places (1 %).
        """
        epsilon = 1e-10  # a small number for preventing indeterminate divisions
        max_abs_diff = np.max(np.abs(cpu_value - gpu_value))
        relative_max_abs_diff_pct = (
            np.max(np.abs((cpu_value - gpu_value) / (epsilon + cpu_value))) * 100.0
        )
        # Assert that the max absolute difference is smaller than the threshold
        # or the relative_max_abs_diff_pct is smaller (when the values are high)
        assert (
            max_abs_diff < threshold_pct / 100.0
            or relative_max_abs_diff_pct < threshold_pct
        ), "There are some inconsistencies between the cpu and gpu values!"
