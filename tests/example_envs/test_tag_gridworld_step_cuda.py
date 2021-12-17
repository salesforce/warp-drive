# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np
import torch

from warp_drive.managers.data_manager import CUDADataManager
from warp_drive.managers.function_manager import (
    CUDAEnvironmentReset,
    CUDAFunctionManager,
    CUDASampler,
)
from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)

_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"
_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS


def cuda_tag_gridworld_step(
    function_manager: CUDAFunctionManager,
    data_manager: CUDADataManager,
    env_resetter: CUDAEnvironmentReset,
):
    # reset env if done flag is seen
    env_resetter.reset_when_done(data_manager)
    tag_gridworld_step = function_manager.get_function("CudaTagGridWorldStep")

    tag_gridworld_step(
        data_manager.device_data("loc_x"),
        data_manager.device_data("loc_y"),
        data_manager.device_data(_ACTIONS),
        data_manager.device_data("_done_"),
        data_manager.device_data(_REWARDS),
        data_manager.device_data(_OBSERVATIONS),
        data_manager.device_data("wall_hit_penalty"),
        data_manager.device_data("tag_reward_for_tagger"),
        data_manager.device_data("tag_penalty_for_runner"),
        data_manager.device_data("step_cost_for_tagger"),
        data_manager.device_data("use_full_observation"),
        data_manager.device_data("world_boundary"),
        data_manager.device_data("_timestep_"),
        data_manager.meta_info("episode_length"),
        block=function_manager.block,
        grid=function_manager.grid,
    )

    ################################################################


class TestCUDAEnvTagGridWorld(unittest.TestCase):
    """
    Unit tests for the (CUDA) step function in Tag GridWorld.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = CUDADataManager(num_agents=5, num_envs=2, episode_length=1)
        self.fm = CUDAFunctionManager(
            num_agents=int(self.dm.meta_info("n_agents")),
            num_envs=int(self.dm.meta_info("n_envs")),
        )
        self.fm.load_cuda_from_binary_file(f"{_CUBIN_FILEPATH}/test_build.fatbin")
        self.resetter = CUDAEnvironmentReset(function_manager=self.fm)
        self.sampler = CUDASampler(function_manager=self.fm)
        self.sampler.init_random(seed=None)

        self.int_step()

    def int_step(self):

        wall_hit_penalty = 0.1
        tag_reward_for_tagger = 10.0
        tag_penalty_for_runner = 2.0
        step_cost_for_tagger = 0.01
        world_boundary = 4
        kIndexToActionArr = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]

        self.fm.initialize_functions(["CudaTagGridWorldStep"])

        # no-op [0, 0], right [1, 0], left [-1, 0], up [0, 1], down [0, -1]
        self.dm.add_shared_constants({"kIndexToActionArr": kIndexToActionArr})
        self.fm.initialize_shared_constants(
            self.dm, constant_names=["kIndexToActionArr"]
        )

        data = DataFeed()
        data.add_data(name="wall_hit_penalty", data=wall_hit_penalty)
        data.add_data(name="tag_reward_for_tagger", data=tag_reward_for_tagger)
        data.add_data(name="tag_penalty_for_runner", data=tag_penalty_for_runner)
        data.add_data(name="step_cost_for_tagger", data=step_cost_for_tagger)
        data.add_data(name="world_boundary", data=world_boundary)
        data.add_data(name="use_full_observation", data=True)
        # states is of the shape (num_envs, num_agents,) = (2, 5,)
        data.add_data(
            name="loc_x",
            data=np.array([[0, 0, 0, 0, 1], [1, 3, 3, 1, 0]]),
            save_copy_and_apply_at_reset=True,
        )
        data.add_data(
            name="loc_y",
            data=np.array([[0, 0, 0, 0, 1], [1, 1, 3, 3, 1]]),
            save_copy_and_apply_at_reset=True,
        )
        self.dm.push_data_to_device(data)
        tensor = DataFeed()
        # rewards is of the shape (num_envs, num_agents) = (2, 5)
        tensor.add_data(
            name=_REWARDS,
            data=np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]]),
        )
        # observation is shaped (num_envs, num_agents, 4*num_agents+1) = (2, 5, 21)
        tensor.add_data(name=_OBSERVATIONS, data=np.zeros((2, 5, 21), dtype=np.float32))
        tensor.add_data(
            name=_ACTIONS, data=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        )
        self.dm.push_data_to_device(tensor, torch_accessible=True)

    def test_step(self):
        # first step, assume got the spread agent distribution
        # and adversary distribution
        # (used 100%, so it is fixed for testing)
        agent_distribution = np.array(
            [
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                ],
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        agent_distribution = torch.from_numpy(agent_distribution)
        agent_distribution = agent_distribution.float().cuda()

        self.sampler.register_actions(self.dm, action_name=_ACTIONS, num_actions=5)

        self.sampler.sample(self.dm, agent_distribution, action_name=_ACTIONS)

        # second step, run one-step update
        cuda_tag_gridworld_step(self.fm, self.dm, self.resetter)

        rewards_update = self.dm.pull_data_from_device(_REWARDS)
        observations_update = self.dm.pull_data_from_device(_OBSERVATIONS)
        actions_update = self.dm.pull_data_from_device(_ACTIONS)
        done_update = self.dm.pull_data_from_device("_done_")

        ref_rewards = np.array(
            [[9.9, 10.0, 10.0, 9.9, -2.0], [-0.01, -0.01, -0.01, -0.01, 0.01]]
        )

        ref_observations = np.array(
            [
                [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                    ],
                ],
                [
                    [
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                    ],
                    [
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        1.0,
                        1.0,
                        2.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                    ],
                ],
            ]
        )

        ref_actions = np.array([[2, 0, 3, 4, 2], [1, 3, 2, 4, 1]])

        self.assertTrue(np.absolute(rewards_update - ref_rewards).max() < 1e-5)
        self.assertTrue(
            np.absolute(
                observations_update.reshape(*observations_update.shape[:2], -1) * 4
                - ref_observations
            ).max()
            < 1e-5
        )
        self.assertTrue(np.absolute(actions_update - ref_actions).max() < 1e-5)
        self.assertSequenceEqual(list(done_update), [1, 1])

        # third step, the reset should happen
        # assume got another spread agent distribution and adversary distribution

        agent_distribution = np.array(
            [
                [
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                ],
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
        agent_distribution = torch.from_numpy(agent_distribution)
        agent_distribution = agent_distribution.float().cuda()

        self.sampler.sample(self.dm, agent_distribution, action_name=_ACTIONS)

        cuda_tag_gridworld_step(self.fm, self.dm, self.resetter)

        rewards_update = self.dm.pull_data_from_device(_REWARDS)
        observations_update = self.dm.pull_data_from_device(_OBSERVATIONS)
        actions_update = self.dm.pull_data_from_device(_ACTIONS)
        done_update = self.dm.pull_data_from_device("_done_")

        ref_rewards = np.array(
            [[-0.01, -0.11, -0.01, -0.11, 0.01], [10.0, 10.0, 10.0, 10.0, -2.0]]
        )

        ref_observations = np.array(
            [
                [
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                    ],
                    [
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                    ],
                ],
                [
                    [
                        0.0,
                        4.0,
                        3.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        4.0,
                        3.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        4.0,
                        3.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        4.0,
                        3.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        4.0,
                    ],
                    [
                        0.0,
                        4.0,
                        3.0,
                        1.0,
                        0.0,
                        1.0,
                        1.0,
                        3.0,
                        2.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        4.0,
                        4.0,
                    ],
                ],
            ]
        )

        ref_actions = np.array([[1, 2, 3, 4, 3], [2, 1, 0, 4, 0]])

        self.assertTrue(np.absolute(rewards_update - ref_rewards).max() < 1e-5)
        self.assertTrue(
            np.absolute(
                observations_update.reshape(*observations_update.shape[:2], -1) * 4
                - ref_observations
            ).max()
            < 1e-5
        )
        self.assertTrue(np.absolute(actions_update - ref_actions).max() < 1e-5)
        self.assertSequenceEqual(list(done_update), [1, 1])
