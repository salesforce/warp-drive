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
    NumbaEnvironmentReset,
    NumbaFunctionManager,
)
from warp_drive.utils.common import get_project_root
from warp_drive.utils.data_feed import DataFeed

_NUMBA_FILEPATH = f"warp_drive.numba_includes"


class TestEnvironmentReset(unittest.TestCase):
    """
    Unit tests for the CUDA environment resetter
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = NumbaDataManager(num_agents=5, num_envs=2, episode_length=2)
        self.fm = NumbaFunctionManager(
            num_agents=int(self.dm.meta_info("n_agents")),
            num_envs=int(self.dm.meta_info("n_envs")),
        )
        self.fm.import_numba_from_source_code(f"{_NUMBA_FILEPATH}.test_build")
        self.resetter = NumbaEnvironmentReset(function_manager=self.fm)
        self.resetter.init_reset_pool(self.dm)

    def test_reset_for_different_dim(self):

        self.dm.data_on_device_via_torch("_done_")[:] = torch.from_numpy(
            np.array([1, 0])
        ).cuda()

        done = self.dm.pull_data_from_device("_done_")
        self.assertSequenceEqual(list(done), [1, 0])

        # expected mean would be around 0.5 * (1+2+3+15) / 4 = 2.625
        a_reset_pool = np.random.rand(4, 10, 3)
        a_reset_pool[1] *= 2
        a_reset_pool[2] *= 3
        a_reset_pool[3] *= 15

        data_feed = DataFeed()
        data_feed.add_data(
            name="a", data=np.random.randn(2, 10, 3), save_copy_and_apply_at_reset=False
        )
        data_feed.add_pool_for_reset(name="a_reset_pool", data=a_reset_pool, reset_target="a")

        self.dm.push_data_to_device(data_feed)

        a = self.dm.pull_data_from_device("a")

        # soft reset
        a_after_reset_0_mean = []
        a_after_reset_1_mean = []
        for _ in range(5000):
            self.resetter.reset_when_done(self.dm, mode="if_done", undo_done_after_reset=False)
            a_after_reset = self.dm.pull_data_from_device("a")
            a_after_reset_0_mean.append(a_after_reset[0].mean())
            a_after_reset_1_mean.append(a_after_reset[1].mean())
        # env 0 should have 1000 times random reset from the pool, so it should close to a_reset_pool.mean()
        self.assertTrue(np.absolute(a_reset_pool.mean() - np.array(a_after_reset_0_mean).mean()) < 1e-1)
        print(a_reset_pool.mean())
        print(np.array(a_after_reset_0_mean).mean())
        # env 1 has no reset at all, so it should be exactly the same as the original one
        self.assertTrue(np.absolute(a[1].mean() - np.array(a_after_reset_1_mean).mean()) < 1e-5)

        # hard reset
        a_after_reset_0_mean = []
        a_after_reset_1_mean = []
        for _ in range(5000):
            self.resetter.reset_when_done(self.dm, mode="force_reset", undo_done_after_reset=False)
            a_after_reset = self.dm.pull_data_from_device("a")
            a_after_reset_0_mean.append(a_after_reset[0].mean())
            a_after_reset_1_mean.append(a_after_reset[1].mean())
        # env 0 should have 1000 times random reset from the pool, so it should close to a_reset_pool.mean()
        self.assertTrue(np.absolute(a_reset_pool.mean() - np.array(a_after_reset_0_mean).mean()) < 1e-1)
        # env 1 should have 1000 times random reset from the pool, so it should close to a_reset_pool.mean()
        self.assertTrue(np.absolute(a_reset_pool.mean() - np.array(a_after_reset_1_mean).mean()) < 1e-1)




