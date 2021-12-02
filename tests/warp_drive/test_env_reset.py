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
)
from warp_drive.utils.common import get_project_root
from warp_drive.utils.data_feed import DataFeed

pytorch_cuda_init_success = torch.cuda.FloatTensor(8)

_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"


class TestEnvironmentReset(unittest.TestCase):
    """
    Unit tests for the CUDA environment resetter
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = CUDADataManager(num_agents=5, num_envs=2, episode_length=2)
        self.fm = CUDAFunctionManager(
            num_agents=int(self.dm.meta_info("n_agents")),
            num_envs=int(self.dm.meta_info("n_envs")),
        )
        self.fm.load_cuda_from_binary_file(f"{_CUBIN_FILEPATH}/test_build.fatbin")
        self.resetter = CUDAEnvironmentReset(function_manager=self.fm)

    def test_reset_for_different_dim(self):

        self.dm.data_on_device_via_torch("_done_")[:] = torch.from_numpy(
            np.array([1, 0])
        ).cuda()

        done = self.dm.pull_data_from_device("_done_")
        self.assertSequenceEqual(list(done), [1, 0])

        data_feed = DataFeed()
        data_feed.add_data(
            name="a", data=np.random.randn(2, 10, 3), save_copy_and_apply_at_reset=True
        )
        data_feed.add_data(
            name="b", data=np.random.randn(2, 10), save_copy_and_apply_at_reset=True
        )
        data_feed.add_data(
            name="c", data=np.random.randn(2), save_copy_and_apply_at_reset=True
        )
        data_feed.add_data(
            name="d",
            data=np.random.randint(10, size=(2, 10, 3), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        data_feed.add_data(
            name="e",
            data=np.random.randint(10, size=(2, 10), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        data_feed.add_data(
            name="f",
            data=np.random.randint(10, size=2, dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )

        self.dm.push_data_to_device(data_feed)

        torch_data_feed = DataFeed()
        torch_data_feed.add_data(
            name="at", data=np.random.randn(2, 10, 3), save_copy_and_apply_at_reset=True
        )
        torch_data_feed.add_data(
            name="bt", data=np.random.randn(2, 10), save_copy_and_apply_at_reset=True
        )
        torch_data_feed.add_data(
            name="ct", data=np.random.randn(2), save_copy_and_apply_at_reset=True
        )
        torch_data_feed.add_data(
            name="dt",
            data=np.random.randint(10, size=(2, 10, 3), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        torch_data_feed.add_data(
            name="et",
            data=np.random.randint(10, size=(2, 10), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        torch_data_feed.add_data(
            name="ft",
            data=np.random.randint(10, size=2, dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        self.dm.push_data_to_device(torch_data_feed, torch_accessible=True)

        a = self.dm.pull_data_from_device("a")
        b = self.dm.pull_data_from_device("b")
        c = self.dm.pull_data_from_device("c")
        d = self.dm.pull_data_from_device("d")
        e = self.dm.pull_data_from_device("e")
        f = self.dm.pull_data_from_device("f")
        at = self.dm.pull_data_from_device("at")
        bt = self.dm.pull_data_from_device("bt")
        ct = self.dm.pull_data_from_device("ct")
        dt = self.dm.pull_data_from_device("dt")
        et = self.dm.pull_data_from_device("et")
        ft = self.dm.pull_data_from_device("ft")

        # change the value in place
        self.dm.data_on_device_via_torch("at")[:] = torch.rand(2, 10, 3).cuda()
        self.dm.data_on_device_via_torch("bt")[:] = torch.rand(2, 10).cuda()
        self.dm.data_on_device_via_torch("ct")[:] = torch.rand(2).cuda()
        self.dm.data_on_device_via_torch("dt")[:] = torch.randint(
            10, size=(2, 10, 3)
        ).cuda()
        self.dm.data_on_device_via_torch("et")[:] = torch.randint(
            10, size=(2, 10)
        ).cuda()
        self.dm.data_on_device_via_torch("ft")[:] = torch.randint(10, size=(2,)).cuda()

        self.resetter.reset_when_done(self.dm)

        a_after_reset = self.dm.pull_data_from_device("a")
        b_after_reset = self.dm.pull_data_from_device("b")
        c_after_reset = self.dm.pull_data_from_device("c")
        d_after_reset = self.dm.pull_data_from_device("d")
        e_after_reset = self.dm.pull_data_from_device("e")
        f_after_reset = self.dm.pull_data_from_device("f")

        at_after_reset = self.dm.pull_data_from_device("at")
        bt_after_reset = self.dm.pull_data_from_device("bt")
        ct_after_reset = self.dm.pull_data_from_device("ct")
        dt_after_reset = self.dm.pull_data_from_device("dt")
        et_after_reset = self.dm.pull_data_from_device("et")
        ft_after_reset = self.dm.pull_data_from_device("ft")

        self.assertTrue(np.absolute((a - a_after_reset).mean()) < 1e-5)
        self.assertTrue(np.absolute((b - b_after_reset).mean()) < 1e-5)
        self.assertTrue(np.absolute((c - c_after_reset).mean()) < 1e-5)
        self.assertTrue(np.count_nonzero(d - d_after_reset) == 0)
        self.assertTrue(np.count_nonzero(e - e_after_reset) == 0)
        self.assertTrue(np.count_nonzero(f - f_after_reset) == 0)

        # so after the soft reset, only env_0 got reset because it has done flag on
        self.assertTrue(np.absolute((at - at_after_reset)[0].mean()) < 1e-5)
        self.assertTrue(np.absolute((bt - bt_after_reset)[0].mean()) < 1e-5)
        self.assertTrue(np.absolute((ct - ct_after_reset)[0].mean()) < 1e-5)
        self.assertTrue(np.absolute((at - at_after_reset)[1].mean()) > 1e-5)
        self.assertTrue(np.absolute((bt - bt_after_reset)[1].mean()) > 1e-5)
        self.assertTrue(np.absolute((ct - ct_after_reset)[1].mean()) > 1e-5)
        self.assertTrue(np.count_nonzero((dt - dt_after_reset)[0]) == 0)
        self.assertTrue(np.count_nonzero((et - et_after_reset)[0]) == 0)
        self.assertTrue(np.count_nonzero((ft - ft_after_reset)[0]) == 0)
        self.assertTrue(np.count_nonzero((dt - dt_after_reset)[1]) > 0)
        self.assertTrue(np.count_nonzero((et - et_after_reset)[1]) > 0)
        self.assertTrue(np.count_nonzero((ft - ft_after_reset)[1]) >= 0)

        done = self.dm.pull_data_from_device("_done_")
        self.assertSequenceEqual(list(done), [0, 0])

        # Now test if mode="force_reset" works

        torch_data_feed2 = DataFeed()
        torch_data_feed2.add_data(
            name="af", data=np.random.randn(2, 10, 3), save_copy_and_apply_at_reset=True
        )
        torch_data_feed2.add_data(
            name="bf", data=np.random.randn(2, 10), save_copy_and_apply_at_reset=True
        )
        torch_data_feed2.add_data(
            name="cf", data=np.random.randn(2), save_copy_and_apply_at_reset=True
        )
        torch_data_feed2.add_data(
            name="df",
            data=np.random.randint(10, size=(2, 10, 3), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        torch_data_feed2.add_data(
            name="ef",
            data=np.random.randint(10, size=(2, 10), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        torch_data_feed2.add_data(
            name="ff",
            data=np.random.randint(10, size=2, dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        self.dm.push_data_to_device(torch_data_feed2, torch_accessible=True)

        af = self.dm.pull_data_from_device("af")
        bf = self.dm.pull_data_from_device("bf")
        cf = self.dm.pull_data_from_device("cf")
        df = self.dm.pull_data_from_device("df")
        ef = self.dm.pull_data_from_device("ef")
        ff = self.dm.pull_data_from_device("ff")

        # change the value in place
        self.dm.data_on_device_via_torch("af")[:] = torch.rand(2, 10, 3).cuda()
        self.dm.data_on_device_via_torch("bf")[:] = torch.rand(2, 10).cuda()
        self.dm.data_on_device_via_torch("cf")[:] = torch.rand(2).cuda()
        self.dm.data_on_device_via_torch("df")[:] = torch.randint(
            10, size=(2, 10, 3)
        ).cuda()
        self.dm.data_on_device_via_torch("ef")[:] = torch.randint(
            10, size=(2, 10)
        ).cuda()
        self.dm.data_on_device_via_torch("ff")[:] = torch.randint(10, size=(2,)).cuda()

        self.resetter.reset_when_done(self.dm)

        af_after_soft_reset = self.dm.pull_data_from_device("af")
        bf_after_soft_reset = self.dm.pull_data_from_device("bf")
        cf_after_soft_reset = self.dm.pull_data_from_device("cf")
        df_after_soft_reset = self.dm.pull_data_from_device("df")
        ef_after_soft_reset = self.dm.pull_data_from_device("ef")
        ff_after_soft_reset = self.dm.pull_data_from_device("ff")

        self.assertTrue(np.absolute((af - af_after_soft_reset).mean()) > 1e-5)
        self.assertTrue(np.absolute((bf - bf_after_soft_reset).mean()) > 1e-5)
        self.assertTrue(np.absolute((cf - cf_after_soft_reset).mean()) > 1e-5)
        self.assertTrue(np.count_nonzero(df - df_after_soft_reset) > 0)
        self.assertTrue(np.count_nonzero(ef - ef_after_soft_reset) > 0)
        self.assertTrue(np.count_nonzero(ff - ff_after_soft_reset) > 0)

        self.resetter.reset_when_done(self.dm, mode="force_reset")

        af_after_hard_reset = self.dm.pull_data_from_device("af")
        bf_after_hard_reset = self.dm.pull_data_from_device("bf")
        cf_after_hard_reset = self.dm.pull_data_from_device("cf")
        df_after_hard_reset = self.dm.pull_data_from_device("df")
        ef_after_hard_reset = self.dm.pull_data_from_device("ef")
        ff_after_hard_reset = self.dm.pull_data_from_device("ff")

        self.assertTrue(np.absolute((af - af_after_hard_reset).mean()) < 1e-5)
        self.assertTrue(np.absolute((bf - bf_after_hard_reset).mean()) < 1e-5)
        self.assertTrue(np.absolute((cf - cf_after_hard_reset).mean()) < 1e-5)
        self.assertTrue(np.count_nonzero(df - df_after_hard_reset) == 0)
        self.assertTrue(np.count_nonzero(ef - ef_after_hard_reset) == 0)
        self.assertTrue(np.count_nonzero(ff - ff_after_hard_reset) == 0)
