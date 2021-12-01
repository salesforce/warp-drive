# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np
import torch

from warp_drive.managers.data_manager import CUDADataManager
from warp_drive.managers.function_manager import CUDAFunctionManager, CUDASampler
from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)

_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"
_ACTIONS = Constants.ACTIONS


class TestActionSampler(unittest.TestCase):
    """
    Unit tests for the CUDA action sampler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = CUDADataManager(num_agents=5, episode_length=1, num_envs=2)
        self.fm = CUDAFunctionManager(
            num_agents=int(self.dm.meta_info("n_agents")),
            num_envs=int(self.dm.meta_info("n_envs")),
        )
        self.fm.load_cuda_from_binary_file(f"{_CUBIN_FILEPATH}/test_build.fatbin")
        self.sampler = CUDASampler(function_manager=self.fm)
        self.sampler.init_random(seed=None)

    def test_agent_action_distribution(self):

        tensor = DataFeed()
        tensor.add_data(name=f"{_ACTIONS}_a", data=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.dm.push_data_to_device(tensor, torch_accessible=True)
        self.assertTrue(self.dm.is_data_on_device_via_torch(f"{_ACTIONS}_a"))

        self.sampler.register_actions(self.dm, f"{_ACTIONS}_a", 3)

        agent_distribution = np.array(
            [
                [
                    [0.333, 0.333, 0.333],
                    [0.2, 0.5, 0.3],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                [
                    [0.1, 0.7, 0.2],
                    [0.7, 0.2, 0.1],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                ],
            ]
        )
        agent_distribution = torch.from_numpy(agent_distribution)
        agent_distribution = agent_distribution.float().cuda()

        # run 10000 times to collect statistics
        actions_a_cuda = torch.from_numpy(
            np.empty((10000, 2, 5), dtype=np.int32)
        ).cuda()

        for i in range(10000):
            self.sampler.sample(
                self.dm, agent_distribution, action_name=f"{_ACTIONS}_a"
            )
            actions_a_cuda[i] = self.dm.data_on_device_via_torch(f"{_ACTIONS}_a")
        actions_a = actions_a_cuda.cpu().numpy()

        actions_a_env_0 = actions_a[:, 0]
        actions_a_env_1 = actions_a[:, 1]

        # Sampler is based on distribution, we test
        # sample mean = given mean and deviation < 10% mean

        self.assertAlmostEqual(
            (actions_a_env_0[:, 0] == 0).sum() / 10000.0, 0.333, delta=0.03
        )
        self.assertAlmostEqual(
            (actions_a_env_0[:, 0] == 1).sum() / 10000.0, 0.333, delta=0.03
        )
        self.assertAlmostEqual(
            (actions_a_env_0[:, 0] == 2).sum() / 10000.0, 0.333, delta=0.03
        )

        self.assertAlmostEqual(
            (actions_a_env_0[:, 1] == 0).sum() / 10000.0, 0.2, delta=0.02
        )
        self.assertAlmostEqual(
            (actions_a_env_0[:, 1] == 1).sum() / 10000.0, 0.5, delta=0.05
        )
        self.assertAlmostEqual(
            (actions_a_env_0[:, 1] == 2).sum() / 10000.0, 0.3, delta=0.03
        )

        self.assertEqual((actions_a_env_0[:, 2] == 0).sum(), 10000)
        self.assertEqual((actions_a_env_0[:, 3] == 1).sum(), 10000)
        self.assertEqual((actions_a_env_0[:, 4] == 2).sum(), 10000)

        self.assertAlmostEqual(
            (actions_a_env_1[:, 0] == 0).sum() / 10000.0, 0.1, delta=0.01
        )
        self.assertAlmostEqual(
            (actions_a_env_1[:, 0] == 1).sum() / 10000.0, 0.7, delta=0.07
        )
        self.assertAlmostEqual(
            (actions_a_env_1[:, 0] == 2).sum() / 10000.0, 0.2, delta=0.02
        )

        self.assertAlmostEqual(
            (actions_a_env_1[:, 1] == 0).sum() / 10000.0, 0.7, delta=0.07
        )
        self.assertAlmostEqual(
            (actions_a_env_1[:, 1] == 1).sum() / 10000.0, 0.2, delta=0.02
        )
        self.assertAlmostEqual(
            (actions_a_env_1[:, 1] == 2).sum() / 10000.0, 0.1, delta=0.01
        )

        self.assertAlmostEqual(
            (actions_a_env_1[:, 2] == 0).sum() / 10000.0, 0.5, delta=0.05
        )
        self.assertAlmostEqual(
            (actions_a_env_1[:, 2] == 1).sum() / 10000.0, 0.5, delta=0.05
        )
        self.assertEqual((actions_a_env_1[:, 2] == 2).sum(), 0)

        self.assertEqual((actions_a_env_1[:, 3] == 0).sum(), 0)
        self.assertAlmostEqual(
            (actions_a_env_1[:, 3] == 1).sum() / 10000.0, 0.5, delta=0.05
        )
        self.assertAlmostEqual(
            (actions_a_env_1[:, 3] == 2).sum() / 10000.0, 0.5, delta=0.05
        )

        self.assertAlmostEqual(
            (actions_a_env_1[:, 4] == 0).sum() / 10000.0, 0.5, delta=0.05
        )
        self.assertEqual((actions_a_env_1[:, 4] == 1).sum(), 0)
        self.assertAlmostEqual(
            (actions_a_env_1[:, 4] == 2).sum() / 10000.0, 0.5, delta=0.05
        )

    def test_planner_action_distribution(self):

        tensor = DataFeed()
        tensor.add_data(name=f"{_ACTIONS}_p", data=[[0], [0]])
        self.dm.push_data_to_device(tensor, torch_accessible=True)
        self.assertTrue(self.dm.is_data_on_device_via_torch(f"{_ACTIONS}_p"))

        self.sampler.register_actions(self.dm, f"{_ACTIONS}_p", 4)

        planner_distribution = np.array(
            [[[0.25, 0.25, 0.25, 0.25]], [[0.10, 0.60, 0.15, 0.15]]]
        )
        planner_distribution = torch.from_numpy(planner_distribution)
        planner_distribution = planner_distribution.float().cuda()

        # run 10000 times to collect statistics
        actions_p_cuda = torch.from_numpy(
            np.empty((10000, 2, 1), dtype=np.int32)
        ).cuda()
        for i in range(10000):
            self.sampler.sample(
                self.dm, planner_distribution, action_name=f"{_ACTIONS}_p"
            )
            actions_p_cuda[i] = self.dm.data_on_device_via_torch(f"{_ACTIONS}_p")
        actions_p = actions_p_cuda.cpu().numpy()
        actions_p_env_0 = actions_p[:, 0]
        actions_p_env_1 = actions_p[:, 1]
        self.assertAlmostEqual(
            (actions_p_env_0[:, 0] == 0).sum() / 10000.0, 0.25, delta=0.03
        )
        self.assertAlmostEqual(
            (actions_p_env_0[:, 0] == 1).sum() / 10000.0, 0.25, delta=0.03
        )
        self.assertAlmostEqual(
            (actions_p_env_0[:, 0] == 2).sum() / 10000.0, 0.25, delta=0.03
        )
        self.assertAlmostEqual(
            (actions_p_env_0[:, 0] == 3).sum() / 10000.0, 0.25, delta=0.03
        )

        self.assertAlmostEqual(
            (actions_p_env_1[:, 0] == 0).sum() / 10000.0, 0.1, delta=0.01
        )
        self.assertAlmostEqual(
            (actions_p_env_1[:, 0] == 1).sum() / 10000.0, 0.6, delta=0.06
        )
        self.assertAlmostEqual(
            (actions_p_env_1[:, 0] == 2).sum() / 10000.0, 0.15, delta=0.015
        )
        self.assertAlmostEqual(
            (actions_p_env_1[:, 0] == 3).sum() / 10000.0, 0.15, delta=0.015
        )

    def test_seed_randomness_across_threads(self):

        tensor = DataFeed()
        tensor.add_data(name=f"{_ACTIONS}_s", data=[[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        self.dm.push_data_to_device(tensor, torch_accessible=True)
        self.assertTrue(self.dm.is_data_on_device_via_torch(f"{_ACTIONS}_s"))

        self.sampler.register_actions(self.dm, f"{_ACTIONS}_s", 4)

        agent_distribution = np.array(
            [
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
                [
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                    [0.25, 0.25, 0.25, 0.25],
                ],
            ]
        )
        agent_distribution = torch.from_numpy(agent_distribution)
        agent_distribution = agent_distribution.float().cuda()

        # run 10 times to collect statistics
        actions_s_cuda = torch.from_numpy(
            np.empty((10000, 2, 5), dtype=np.int32)
        ).cuda()

        for i in range(10000):
            self.sampler.sample(
                self.dm, agent_distribution, action_name=f"{_ACTIONS}_s"
            )
            actions_s_cuda[i] = self.dm.data_on_device_via_torch(f"{_ACTIONS}_s")
        actions_s = actions_s_cuda.cpu().numpy()
        self.assertTrue(actions_s.std(axis=-1).reshape(-1).mean() > 0.9)
