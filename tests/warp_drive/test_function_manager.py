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
    CUDALogController,
)
from warp_drive.utils.common import get_project_root
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)

_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"
_ACTIONS = Constants.ACTIONS


def cuda_dummy_step(
    function_manager: CUDAFunctionManager,
    data_manager: CUDADataManager,
    env_resetter: CUDAEnvironmentReset,
    target: int,
    step: int,
):

    env_resetter.reset_when_done(data_manager)

    step = np.int32(step)
    target = np.int32(target)
    test_step = function_manager.get_function("testkernel")
    test_step(
        data_manager.device_data("X"),
        data_manager.device_data("Y"),
        data_manager.device_data("_done_"),
        data_manager.device_data(_ACTIONS),
        data_manager.device_data("multiplier"),
        target,
        step,
        data_manager.meta_info("episode_length"),
        block=function_manager.block,
        grid=function_manager.grid,
    )
    ################################################################


class TestFunctionManager(unittest.TestCase):
    """
    Unit tests for the CUDA function manager
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = CUDADataManager(num_agents=5, num_envs=2, episode_length=2)
        self.fm = CUDAFunctionManager(
            num_agents=int(self.dm.meta_info("n_agents")),
            num_envs=int(self.dm.meta_info("n_envs")),
        )
        self.fm.load_cuda_from_binary_file(f"{_CUBIN_FILEPATH}/test_build.fatbin")
        self.dc = CUDALogController(function_manager=self.fm)
        self.resetter = CUDAEnvironmentReset(function_manager=self.fm)

    def test_step(self):
        self.fm.initialize_functions(["testkernel"])

        data = DataFeed()
        data.add_data(
            name="X",
            data=[[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
            save_copy_and_apply_at_reset=True,
            log_data_across_episode=True,
        )

        data.add_data(
            name="Y",
            data=np.array(
                [[6, 7, 8, 9, 10], [1, 2, 3, 4, 5]],
            ),
            save_copy_and_apply_at_reset=True,
            log_data_across_episode=True,
        )
        data.add_data(name="multiplier", data=2.0)

        tensor = DataFeed()
        tensor.add_data(
            name=_ACTIONS,
            data=[
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            ],
        )

        self.dm.push_data_to_device(data)
        self.dm.push_data_to_device(tensor, torch_accessible=True)

        self.assertTrue(self.dm.is_data_on_device("X"))
        self.assertTrue(self.dm.is_data_on_device("Y"))
        self.assertTrue(self.dm.is_data_on_device_via_torch(_ACTIONS))

        # test log mask
        dense_log_mask = self.dm.pull_data_from_device("_log_mask_")
        self.assertSequenceEqual(list(dense_log_mask), [0, 0, 0])

        # t = 0 is reserved for the start point
        self.dc.reset_log(data_manager=self.dm, env_id=0)
        for t in range(1, self.dm.meta_info("episode_length") + 1):
            cuda_dummy_step(
                function_manager=self.fm,
                data_manager=self.dm,
                env_resetter=self.resetter,
                target=100,
                step=t,
            )
            self.dc.update_log(data_manager=self.dm, step=t)
        dense_log = self.dc.fetch_log(data_manager=self.dm, names=["X", "Y"])
        # test after two steps, the X and Y log updating
        X_update = dense_log["X_for_log"]
        Y_update = dense_log["Y_for_log"]

        self.assertAlmostEqual(X_update[1].mean(), 0.15, places=6)
        self.assertAlmostEqual(X_update[2].mean(), 0.075, places=6)
        self.assertEqual(Y_update[1].mean(), 16)
        self.assertEqual(Y_update[2].mean(), 32)

        # test action tensor updating
        actions = self.dm.pull_data_from_device(_ACTIONS)

        self.assertEqual(actions[0, :, 0].mean(), 0)
        self.assertEqual(actions[0, :, 1].mean(), 1)
        self.assertEqual(actions[0, :, 2].mean(), 2)
        self.assertEqual(actions[1, :, 0].mean(), 0)
        self.assertEqual(actions[1, :, 1].mean(), 1)
        self.assertEqual(actions[1, :, 2].mean(), 2)

        # right at this point, the reset functions have not been activated yet,
        # done flags should be all 1's

        done = self.dm.pull_data_from_device("_done_")
        self.assertSequenceEqual(list(done), [1, 1])

        # explicit reset (dummy step function actually will do this,
        # but for testing purpose, call this directly to see)
        self.resetter.reset_when_done(data_manager=self.dm)

        done = self.dm.pull_data_from_device("_done_")
        self.assertSequenceEqual(list(done), [0, 0])

        X_after_reset = self.dm.pull_data_from_device("X")
        Y_after_reset = self.dm.pull_data_from_device("Y")
        # the 0th dim is env
        self.assertAlmostEqual(X_after_reset[0].mean(), 0.3, places=6)
        self.assertAlmostEqual(X_after_reset[1].mean(), 0.8, places=6)
        self.assertEqual(Y_after_reset[0].mean(), 8)
        self.assertEqual(Y_after_reset[1].mean(), 3)

        # test log reset
        self.dc.reset_log(data_manager=self.dm, env_id=1)

        dense_log_mask = self.dm.pull_data_from_device("_log_mask_")
        self.assertSequenceEqual(list(dense_log_mask), [1, 0, 0])
        dense_log = self.dc.fetch_log(data_manager=self.dm, names=["X", "Y"])
        self.assertEqual(len(dense_log["X_for_log"]), 1)
        self.assertEqual(len(dense_log["Y_for_log"]), 1)

        # now run it again, but log env 1 instead
        for t in range(1, self.dm.meta_info("episode_length") + 1):
            cuda_dummy_step(
                function_manager=self.fm,
                data_manager=self.dm,
                env_resetter=self.resetter,
                target=100,
                step=t,
            )
            self.dc.update_log(data_manager=self.dm, step=t)
        dense_log = self.dc.fetch_log(data_manager=self.dm, names=["X", "Y"])
        # test after two steps, the X and Y and the log updating
        X_update = dense_log["X_for_log"]
        Y_update = dense_log["Y_for_log"]

        # the 0th dim is timestep
        self.assertAlmostEqual(X_update[1].mean(), 0.40, places=6)
        self.assertAlmostEqual(X_update[2].mean(), 0.20, places=6)
        self.assertEqual(Y_update[1].mean(), 6)
        self.assertEqual(Y_update[2].mean(), 12)

        # test different env has different reset timing
        # after 1 step, done should be [1, 0] because the first env
        # has several agents reach the target
        step = 1
        cuda_dummy_step(
            function_manager=self.fm,
            data_manager=self.dm,
            env_resetter=self.resetter,
            target=15,
            step=step,
        )
        done = self.dm.pull_data_from_device("_done_")
        self.assertSequenceEqual(list(done), [1, 0])

        # after 2 steps, done should be [1, 1]
        step += 1
        cuda_dummy_step(
            function_manager=self.fm,
            data_manager=self.dm,
            env_resetter=self.resetter,
            target=15,
            step=step,
        )
        done = self.dm.pull_data_from_device("_done_")
        self.assertSequenceEqual(list(done), [1, 1])

        # after 2 steps, env_0 should appear to have 1-time (times 2 or divide 2)
        # update (because of one reset)
        # while env_1 have 2-time (times 4 or divide 4) update
        Y_for_both_envs = self.dm.pull_data_from_device("Y")
        X_for_both_envs = self.dm.pull_data_from_device("X")

        # this is not the log but data itself, the 0th dim is env
        self.assertAlmostEqual(X_for_both_envs[0].mean(), 0.15, places=6)
        self.assertAlmostEqual(X_for_both_envs[1].mean(), 0.20, places=6)
        self.assertEqual(Y_for_both_envs[0].mean(), 16)
        self.assertEqual(Y_for_both_envs[1].mean(), 12)
