# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np
import torch

from warp_drive.managers.numba_managers.numba_data_manager import NumbaDataManager
from warp_drive.managers.numba_managers.numba_function_manager import (
    NumbaFunctionManager,
    NumbaSampler,
)
from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

_NUMBA_FILEPATH = f"warp_drive.numba_includes"
_ACTIONS = Constants.ACTIONS


class TestOUProcessSampler(unittest.TestCase):
    """
    Unit tests for the CUDA action sampler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = NumbaDataManager(num_agents=5, episode_length=1, num_envs=1000)
        self.fm = NumbaFunctionManager(
            num_agents=int(self.dm.meta_info("n_agents")),
            num_envs=int(self.dm.meta_info("n_envs")),
        )
        self.fm.import_numba_from_source_code(f"{_NUMBA_FILEPATH}.test_build")
        self.sampler = NumbaSampler(function_manager=self.fm)
        self.sampler.init_random(seed=None)

    def test_variation(self):
        tensor = DataFeed()
        tensor.add_data(name=f"{_ACTIONS}_a", data=np.zeros((1000, 5, 1), dtype=np.float32))
        self.dm.push_data_to_device(tensor, torch_accessible=True)
        self.sampler.register_actions(self.dm, f"{_ACTIONS}_a", 1, is_deterministic=True)

        # deterministic agent actions
        agent_distribution = np.zeros((1000, 5, 1), dtype=np.float32)
        agent_distribution = torch.from_numpy(agent_distribution)
        agent_distribution = agent_distribution.float().cuda()

        actions_a_cuda = torch.from_numpy(
            np.empty((10000, 1000, 5), dtype=np.float32)
        ).cuda()

        damping = 0.15
        stddev = 0.2
        for i in range(10000):
            self.sampler.sample(self.dm, agent_distribution, action_name=f"{_ACTIONS}_a", damping=damping, stddev=stddev,)
            actions_a_cuda[i] = self.dm.data_on_device_via_torch(f"{_ACTIONS}_a")[:, :, 0]
        actions_a = actions_a_cuda.cpu().numpy()

        var_list = []
        for i in range(100, 10000):
            var_list.append(actions_a[i].flatten().std())
        var_mean = np.array(var_list).mean()

        var_theory = stddev/(1 - (1 - damping)**2)**0.5

        self.assertAlmostEqual(var_mean, var_theory, delta=0.001)

        cov_list = []
        # test the cov of step difference of 10
        # stddev^2/(1-(1-damping)^2)*(1-damping)^(n-k)*[1-(1-damping)^(n+k)]
        # roughly it is stddev^2/(1-(1-damping)^2)*(1-damping)^(n-k)
        for i in range(100, 9990):
            cov_list.append(np.cov(actions_a[i].flatten(), actions_a[i + 10].flatten())[0, 1])
        cov_mean = np.array(cov_list).mean()

        cov_theory = stddev**2 / (1 - (1 - damping)**2) * (1 - damping)**10

        self.assertAlmostEqual(cov_mean, cov_theory, delta=0.001)
