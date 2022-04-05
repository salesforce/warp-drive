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
from warp_drive.training.utils.data_loader import (
    create_and_push_data_placeholders,
    get_obs,
)
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS

_EPSILON = 1e-10  # a small number for preventing indeterminate divisions

# Set logger level e.g., DEBUG, INFO, WARNING, ERROR\n",
logging.getLogger().setLevel(logging.ERROR)


def generate_random_actions(env, num_envs, seed=None):
    """
    Generate random actions for each agent and each env.
    """
    agent_ids = list(env.action_space.keys())
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
        blocks_per_env=None,
        assert_equal_num_agents_per_block=False,
        num_episodes=2,
        use_gpu_testing_mode=False,
        testing_bin_filename=None,
        env_registrar=None,
        env_wrapper=EnvWrapper,
        policy_tag_to_agent_id_map=None,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
    ):
        """
        :param cpu_env_class: cpu env class to test, for example, TagGridWorld
        :param cuda_env_class: cuda env class to test, for example, CUDATagGridWorld
        :param dual_mode_env_class: env class that supports both cpu and cuda
        :param env_config: env configuration
        :param num_envs: number of parallel example_envs in the test.
            If use_gpu_testing_mode = True,
            num_envs = 2 and num_agents=5 are enforced
        :param blocks_per_env: number of blocks to cover one environment
            default is None, the utility function will estimate it
            otherwise it will be reinforced
        :param num_episodes: number of episodes in the test
            hint: number >=2 is recommended
            since it can fully test the reset
        :param use_gpu_testing_mode: a flag to determine whether to simply load
            the cuda binaries (.cubin) or compile the cuda source code (.cu)
            each time to create a binary.`
            Defaults to False.
            If use_gpu_testing_mode is True, do not forget to
            include your testing env into warp_drive/cuda_includes/test_build.cu,
            and the Makefile will automate this build.
        :param testing_bin_filename: load the specified .cubin or .fatbin directly,
            only for use_gpu_testing_mode.
        :param env_registrar: the EnvironmentRegistrar object;
            it provides the customized env info (like src path) for the build
        :param env_wrapper: allows for the user to provide their own EnvWrapper.
            e.g.,https://github.com/salesforce/ai-economist/blob/master/
                 ai_economist/foundation/env_wrapper.py
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
            Defaults to False
        :param obs_dim_corresponding_to_num_agents:
            indicative of which dimension in the observation corresponds
            to the number of agents, as designed in the step function.
            It may be "first" or "last". In other words,
            observations may be shaped (num_agents, *feature_dim) or
            (*feature_dim, num_agents). This is required in order for
            WarpDrive to process the observations correctly. This is only
            relevant when a single obs key corresponds to multiple agents.
            Defaults to "first".
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
        self.blocks_per_env = blocks_per_env
        self.assert_equal_num_agents_per_block = assert_equal_num_agents_per_block
        self.num_episodes = num_episodes
        self.use_gpu_testing_mode = use_gpu_testing_mode
        self.testing_bin_filename = testing_bin_filename
        self.env_registrar = env_registrar
        self.env_wrapper = env_wrapper
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.create_separate_placeholders_for_each_policy = (
            create_separate_placeholders_for_each_policy
        )
        self.obs_dim_corresponding_to_num_agents = obs_dim_corresponding_to_num_agents

    def test_env_reset_and_step(self, consistency_threshold_pct=1, seed=None):
        """
        Perform consistency checks for the reset() and step() functions
        consistency_threshold_pct: consistency threshold as a percentage
        (defaults to 1%).
        """
        for scenario in self.env_configs:
            env_config = self.env_configs[scenario]

            print(f"Performing the consistency checks for scenario: {scenario}...")

            # Env Reset
            # CPU version of env
            # ------------------
            env_cpu = {}
            obs_cpu = []
            for env_id in range(self.num_envs):
                if self.env_registrar is not None and self.env_registrar.has_env(
                    self.cpu_env_class.name, device="cpu"
                ):
                    env_cpu[env_id] = self.env_registrar.get(
                        self.cpu_env_class.name, use_cuda=False
                    )(**env_config)
                else:
                    env_cpu[env_id] = self.env_wrapper(
                        env_obj=self.cpu_env_class(**env_config), use_cuda=False
                    )
                if self.use_gpu_testing_mode:
                    assert env_cpu[env_id].n_agents == 5

                # obs will be a dict of {agent_id: agent_specific_obs}
                # agent_specific_obs could itself be a dictionary or an array
                obs_cpu += [env_cpu[env_id].reset()]

            # GPU version of env
            # ------------------
            if self.env_registrar is not None and self.env_registrar.has_env(
                self.cuda_env_class.name, device="cuda"
            ):
                # we use env_registrar to instantiate env object
                env_obj = None
            else:
                # if not registered, we use the brutal force
                env_obj = self.cuda_env_class(**env_config)
            kwargs = {
                "env_obj": env_obj,
                "env_name": self.cuda_env_class.name,
                "env_config": env_config,
                "num_envs": self.num_envs,
                "blocks_per_env": self.blocks_per_env,
                "use_cuda": True,
                "env_registrar": self.env_registrar,
            }
            # Testing mode
            if self.use_gpu_testing_mode:
                kwargs.update(
                    {
                        "testing_mode": self.use_gpu_testing_mode,
                        "testing_bin_filename": self.testing_bin_filename,
                    }
                )
            env_gpu = self.env_wrapper(**kwargs)
            env_gpu.reset_all_envs()

            # Push obs, sampled actions, rewards and done flags placeholders
            # to the device
            create_and_push_data_placeholders(
                env_gpu,
                self.policy_tag_to_agent_id_map,
                self.create_separate_placeholders_for_each_policy,
                self.obs_dim_corresponding_to_num_agents,
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
            print("Running obs consistency check after the first reset...")
            self._run_obs_consistency_checks(
                obs_cpu, env_gpu, threshold_pct=consistency_threshold_pct, time="reset"
            )
            print("DONE")

            # Consistency checks during subsequent steps and resets
            # -----------------------------------------------------
            print(
                "Running obs/rew/done consistency check "
                "during subsequent env steps and resets..."
            )

            # Test across multiple episodes
            for timestep in range(self.num_episodes * env_gpu.episode_length):
                actions_list_of_dicts = generate_random_actions(
                    env_gpu.env, self.num_envs, seed
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

                # Step through all the environments (CPU)
                obs_list = []
                rew_list = []
                done_list = []

                for env_id in range(self.num_envs):
                    obs, rew, done, _ = env_cpu[env_id].step(
                        actions_list_of_dicts[env_id]
                    )
                    obs_list += [obs]
                    rew_list += [rew]
                    done_list += [done]

                done_cpu = {
                    "__all__": np.array([done["__all__"] for done in done_list])
                }

                # Step through all the environments (GPU)
                env_gpu.step_all_envs()

                done_gpu = (
                    env_gpu.cuda_data_manager.data_on_device_via_torch("_done_")
                    .cpu()
                    .numpy()
                )

                self._run_obs_consistency_checks(
                    obs_list,
                    env_gpu,
                    threshold_pct=consistency_threshold_pct,
                    time=timestep,
                )
                self._run_rew_consistency_checks(
                    rew_list,
                    env_gpu,
                    threshold_pct=consistency_threshold_pct,
                    time=timestep,
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
                        obs_list,
                        env_gpu,
                        threshold_pct=consistency_threshold_pct,
                        time=timestep,
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
                agent_ids = self.policy_tag_to_agent_id_map[policy]

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

    def _get_cpu_gpu_obs(
        self,
        obs,
        env_gpu,
        policy_tag_to_agent_id_map,
        obs_dim_corresponding_to_num_agents="first",
        suffix="",
    ):
        first_env_id = 0

        if suffix == "":
            agent_ids = sorted(list(obs[first_env_id].keys()))
            first_agent_id = agent_ids[0]
        else:
            pol_tag = suffix.split("_")[1]
            agent_ids = policy_tag_to_agent_id_map[pol_tag]
            first_agent_id = agent_ids[0]

        num_envs = env_gpu.n_envs

        if isinstance(obs[first_env_id][first_agent_id], (list, np.ndarray)):
            agent_obs_for_all_envs = [
                get_obs(obs[env_id], agent_ids, obs_dim_corresponding_to_num_agents)
                for env_id in range(num_envs)
            ]
            obs_cpu = {"obs": np.stack(agent_obs_for_all_envs, axis=0)}
            obs_gpu = {
                "obs": env_gpu.cuda_data_manager.pull_data_from_device(
                    f"{_OBSERVATIONS}" + suffix
                )
            }
        elif isinstance(obs[first_env_id][first_agent_id], dict):
            obs_cpu = {}
            obs_gpu = {}
            for key in obs[first_env_id][first_agent_id]:
                agent_obs_for_all_envs = [
                    get_obs(
                        obs[env_id],
                        agent_ids,
                        obs_dim_corresponding_to_num_agents,
                        key=key,
                    )
                    for env_id in range(num_envs)
                ]
                obs_cpu[key] = np.stack(agent_obs_for_all_envs, axis=0)
                obs_gpu[key] = env_gpu.cuda_data_manager.pull_data_from_device(
                    f"{_OBSERVATIONS}" + suffix + f"_{key}"
                )

        else:
            raise NotImplementedError(
                "Only array or dict type observations are supported!"
            )

        assert isinstance(obs_cpu, dict)
        assert isinstance(obs_gpu, dict)
        assert sorted(list(obs_cpu.keys())) == sorted(list(obs_gpu.keys()))
        return obs_cpu, obs_gpu

    def _run_obs_consistency_checks(self, obs, env_gpu, threshold_pct, time=None):
        assert time is not None
        if self.create_separate_placeholders_for_each_policy:
            assert len(self.policy_tag_to_agent_id_map) > 1
            for pol_mod_tag in self.policy_tag_to_agent_id_map:
                suffix = f"_{pol_mod_tag}"
                obs_cpu, obs_gpu = self._get_cpu_gpu_obs(
                    obs,
                    env_gpu,
                    self.policy_tag_to_agent_id_map,
                    self.obs_dim_corresponding_to_num_agents,
                    suffix=suffix,
                )
                for key, val in obs_cpu.items():
                    self._run_consistency_checks(
                        val,
                        obs_gpu[key],
                        threshold_pct,
                        time=time,
                        key=f"observation{suffix} ({key})",
                    )
        else:
            obs_cpu, obs_gpu = self._get_cpu_gpu_obs(
                obs,
                env_gpu,
                self.policy_tag_to_agent_id_map,
                self.obs_dim_corresponding_to_num_agents,
            )
            for key, val in obs_cpu.items():
                self._run_consistency_checks(
                    val,
                    obs_gpu[key],
                    threshold_pct,
                    time=time,
                    key=f"observation ({key})",
                )

    def _get_cpu_gpu_rew(self, rew, env_gpu, policy_tag_to_agent_id_map, suffix=""):
        if suffix == "":
            agent_ids = list(range(env_gpu.n_agents))
            first_agent_id = agent_ids[0]
        else:
            pol_tag = suffix.split("_")[1]
            agent_ids = policy_tag_to_agent_id_map[pol_tag]
            first_agent_id = agent_ids[0]

        num_envs = env_gpu.n_envs
        first_env_id = 0

        assert isinstance(
            rew[first_env_id][first_agent_id], (float, int, np.floating, np.integer)
        )

        rew_cpu = np.stack(
            [
                np.array([rew[env_id][agent_id] for agent_id in agent_ids])
                for env_id in range(num_envs)
            ],
            axis=0,
        )
        rew_gpu = env_gpu.cuda_data_manager.pull_data_from_device(
            f"{_REWARDS}" + suffix
        )

        return rew_cpu, rew_gpu

    def _run_rew_consistency_checks(self, rew, env_gpu, threshold_pct, time=None):
        assert time is not None
        if self.create_separate_placeholders_for_each_policy:
            assert len(self.policy_tag_to_agent_id_map) > 1
            for pol_mod_tag in self.policy_tag_to_agent_id_map:
                rew_cpu, rew_gpu = self._get_cpu_gpu_rew(
                    rew,
                    env_gpu,
                    self.policy_tag_to_agent_id_map,
                    suffix=f"_{pol_mod_tag}",
                )
                self._run_consistency_checks(
                    rew_cpu,
                    rew_gpu,
                    threshold_pct,
                    time=time,
                    key="reward",
                )
        else:
            rew_cpu, rew_gpu = self._get_cpu_gpu_rew(
                rew, env_gpu, self.policy_tag_to_agent_id_map
            )
            self._run_consistency_checks(
                rew_cpu,
                rew_gpu,
                threshold_pct,
                time=time,
                key="reward",
            )

    @staticmethod
    def _run_consistency_checks(
        cpu_value, gpu_value, threshold_pct=1, time=None, key=None
    ):
        """
        Perform consistency checks between the cpu and gpu values.
        The default threshold is 2 decimal places (1 %).
        """
        assert time is not None
        assert key is not None
        abs_diff = np.abs(cpu_value - gpu_value)
        relative_abs_diff_pct = (
            np.abs((cpu_value - gpu_value) / (_EPSILON + cpu_value))
        ) * 100.0
        # Assert that the absolute difference is smaller than the threshold
        # or the relative absolute difference percentage is smaller than the threshold
        # (when the values are high)
        is_consistent = np.logical_or(
            abs_diff < threshold_pct / 100.0, relative_abs_diff_pct < threshold_pct
        )
        try:
            assert is_consistent.all()
        except AssertionError as e:
            mismatched_indices = np.where(np.logical_not(is_consistent))
            for index in zip(*mismatched_indices):
                env_idx = index[0]
                agent_idx = index[1]
                feature_idx = index[2:]
                logging.error(
                    f"Discrepancy found at timestep {time} in {key} "
                    f"for env index: {env_idx}, "
                    f"agent index: {agent_idx} & "
                    f"feature index: {feature_idx};\n"
                    f"cpu(gpu) value: {cpu_value[index]}({gpu_value[index]})\n"
                )
            raise AssertionError(
                "There are some inconsistencies between the cpu and gpu values!"
            ) from e
