# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
Consistency tests for comparing the cuda (gpu) / no cuda (cpu) version
"""

import numpy as np
import torch
from gym.spaces import Discrete, MultiDiscrete

from warp_drive.env_wrapper import EnvWrapper
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS


def generate_random_actions(env, num_envs):
    """
    Generate random actions for each agent and each env.
    """

    assert isinstance(
        env.action_space[0], (Discrete, MultiDiscrete)
    ), "Unknown action space for env."
    if isinstance(env.action_space[0], Discrete):
        return [
            {
                agent_id: np.random.randint(
                    low=0, high=int(env.action_space[agent_id].n), dtype=np.int32
                )
                for agent_id in range(env.num_agents)
            }
            for _ in range(num_envs)
        ]
    return [
        {
            agent_id: env.np_random.randint(
                low=[0] * len(env.action_space[agent_id].nvec),
                high=env.action_space[agent_id].nvec,
                dtype=np.int32,
            )
            for agent_id in range(env.num_agents)
        }
        for _ in range(num_envs)
    ]


class EnvironmentCPUvsGPU:
    """
    test the rollout consistency between the CPU environment and the GPU environment
    """

    def __init__(
        self,
        env_class,
        env_configs,
        num_envs=2,
        num_episodes=2,
        use_gpu_testing_mode=True,
    ):
        """
        :param env_class: env class to test, for example, TagGridWorld
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

        """
        self.env_class = env_class
        self.env_configs = env_configs
        if use_gpu_testing_mode:
            print(
                f"enforce the num_envs = {num_envs} because you have "
                f"use_gpu_testing_mode = True, where the cubin file"
                f"supporting this testing mode assumes 2 parallel example_envs"
            )
        self.num_envs = num_envs
        self.num_episodes = num_episodes
        self.use_gpu_testing_mode = use_gpu_testing_mode

    def test_env_reset_and_step(self):
        """
        Perform consistency checks for the reset() and step() functions
        """

        for key in self.env_configs:
            env_config = self.env_configs[key]

            print(f"Running {key}...")
            # Env Reset
            # CPU version of env
            env_cpu = {}
            obs_list = []

            for env_id in range(self.num_envs):
                env_cpu[env_id] = self.env_class(**env_config)
                if self.use_gpu_testing_mode:
                    assert env_cpu[env_id].num_agents == 5
                # obs will be a dict of {agent_id: agent_obs}
                obs = env_cpu[env_id].reset()

                # Combine obs across agents
                combined_obs_array = np.stack(
                    [obs[key] for key in sorted(obs.keys())], axis=0
                )
                obs_list += [combined_obs_array]

            obs_cpu = np.stack((obs_list), axis=0)

            # GPU version of env
            env_gpu = EnvWrapper(
                self.env_class(**env_config),
                num_envs=self.num_envs,
                use_cuda=True,
                testing_mode=self.use_gpu_testing_mode,
            )
            env_gpu.reset_all_envs()

            # Observations, actions and rewards placeholders
            # ----------------------------------------------

            data_feed = DataFeed()
            data_feed.add_data(
                name=_OBSERVATIONS,
                data=np.stack(
                    [combined_obs_array for _ in range(self.num_envs)], axis=0
                ),
                save_copy_and_apply_at_reset=True,
            )
            data_feed.add_data(
                name=_REWARDS,
                data=np.zeros(
                    (self.num_envs, env_gpu.env.num_agents), dtype=np.float32
                ),
            )
            env_gpu.cuda_data_manager.push_data_to_device(
                data_feed, torch_accessible=True
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

            # Consistency checks at the first reset
            # -------------------------------------
            obs_gpu = env_gpu.cuda_data_manager.pull_data_from_device(_OBSERVATIONS)
            print("Running obs consistency check after first reset...")
            self.run_consistency_checks(obs_cpu, obs_gpu)

            # Consistency checks during step
            # ------------------------------
            print("Running obs/rew/done consistency check during env resets and steps")

            # Test across multiple episodes
            for _ in range(self.num_episodes * env_config["episode_length"]):
                actions_list_of_dicts = generate_random_actions(
                    env_cpu[env_id], self.num_envs
                )

                actions_list = []
                for action_dict in actions_list_of_dicts:
                    combined_actions = np.stack(
                        [action_dict[key] for key in sorted(action_dict.keys())], axis=0
                    )
                    actions_list += [combined_actions]

                actions = np.stack((actions_list), axis=0)

                actions_data = DataFeed()
                actions_data.add_data(name=_ACTIONS, data=actions)

                if not env_gpu.cuda_data_manager.is_data_on_device_via_torch(_ACTIONS):
                    env_gpu.cuda_data_manager.push_data_to_device(
                        actions_data, torch_accessible=True
                    )
                else:
                    env_gpu.cuda_data_manager.data_on_device_via_torch(_ACTIONS)[
                        :
                    ] = torch.from_numpy(actions)

                obs_list = []
                rew_list = []
                done_list = []

                for env_id in range(self.num_envs):
                    obs, rew, done, _ = env_cpu[env_id].step(
                        actions_list_of_dicts[env_id]
                    )

                    # Combine obs across agents
                    combined_obs_array = np.stack(
                        [obs[key] for key in sorted(obs.keys())], axis=0
                    )
                    obs_list += [combined_obs_array]

                    combined_rew_array = np.stack(
                        [rew[key] for key in sorted(rew.keys())], axis=0
                    )
                    rew_list += [combined_rew_array]
                    done_list += [done]

                obs_cpu = np.stack((obs_list), axis=0)
                rew_cpu = np.stack((rew_list), axis=0)

                done_cpu = {
                    "__all__": np.array([done["__all__"] for done in done_list])
                }

                # Update actions tensor on the gpu
                _, _, done_gpu, _ = env_gpu.step()
                done_gpu["__all__"] = done_gpu["__all__"].cpu().numpy()

                obs_gpu = env_gpu.cuda_data_manager.pull_data_from_device(_OBSERVATIONS)
                rew_gpu = env_gpu.cuda_data_manager.pull_data_from_device(_REWARDS)

                self.run_consistency_checks(obs_cpu, obs_gpu)
                self.run_consistency_checks(rew_cpu, rew_gpu)
                assert all(done_cpu["__all__"] == done_gpu["__all__"])

                # GPU reset
                env_gpu.reset_only_done_envs()

                # Now, pull done flags and they should be set to 0 (False) again
                done_gpu = env_gpu.cuda_data_manager.pull_data_from_device("_done_")
                assert done_gpu.sum() == 0

                obs_gpu = env_gpu.cuda_data_manager.pull_data_from_device(_OBSERVATIONS)
                # CPU reset
                for env_id in range(self.num_envs):
                    if done_cpu["__all__"][env_id]:
                        # Reset the CPU for this env_id
                        obs = env_cpu[env_id].reset()

                        # Combine obs across agents
                        obs_cpu = np.stack(
                            [obs[key] for key in sorted(obs.keys())], axis=0
                        )

                        # Run obs consistency checks at reset
                        self.run_consistency_checks(obs_cpu, obs_gpu[env_id])

    @staticmethod
    def run_consistency_checks(cpu_value, gpu_value, decimal_places=3):
        """
        Perform consistency checks between the cpu and gpu values.
        The default threshold is 3 decimal places.
        """
        assert np.max(np.abs(cpu_value - gpu_value)) < 10 ** (-decimal_places)
