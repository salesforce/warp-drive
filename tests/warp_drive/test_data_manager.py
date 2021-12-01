# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np

from warp_drive.managers.data_manager import CUDADataManager
from warp_drive.utils.data_feed import DataFeed


class TestDataManager(unittest.TestCase):
    """
    Unit tests for the CUDA data manager
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = CUDADataManager(num_agents=5, num_envs=1, episode_length=3)

    def test_add_meta_info(self):
        self.dm.add_meta_info(meta={"learning_rate": 0.01})
        self.assertEqual(self.dm.meta_info("n_agents"), 5)
        self.assertTrue(isinstance(self.dm.meta_info("n_agents"), np.int32))
        self.assertEqual(self.dm.meta_info("episode_length"), 3)
        self.assertTrue(isinstance(self.dm.meta_info("episode_length"), np.int32))
        self.assertEqual(self.dm.meta_info("n_envs"), 1)
        self.assertTrue(isinstance(self.dm.meta_info("n_envs"), np.int32))
        self.assertAlmostEqual(self.dm.meta_info("learning_rate"), 0.01, places=6)
        self.assertTrue(isinstance(self.dm.meta_info("learning_rate"), np.float32))

    def test_add_data_and_push_to_device(self):

        data = DataFeed()
        data.add_data(
            name="X", data=np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
        )
        data.add_data(
            name="Y",
            data=[
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        data.add_data(name="a", data=100)
        data.add_data(name="b", data=1.0)

        self.dm.push_data_to_device(data)
        x = self.dm.pull_data_from_device("X")
        y = self.dm.pull_data_from_device("Y")
        a = self.dm.pull_data_from_device("a")
        b = self.dm.pull_data_from_device("b")
        self.assertEqual(x[0].mean(), 3)
        self.assertEqual(x[1].mean(), 0)
        self.assertEqual(x[2].mean(), 0)
        self.assertAlmostEqual(y[0].mean(), 0.3, places=6)
        self.assertAlmostEqual(y[1].mean(), 0.0, places=6)
        self.assertAlmostEqual(y[2].mean(), 0.0, places=6)
        self.assertEqual(a, 100)
        self.assertEqual(b, 1.0)

    def test_add_tensor_and_push_to_device(self):
        data = DataFeed()
        data.add_data(
            name="Xt",
            data=np.array([[1, 2, 3, 4, 5], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
        )
        data.add_data(
            name="Yt",
            data=[
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        )
        self.dm.push_data_to_device(data, torch_accessible=True)
        xt = self.dm.pull_data_from_device("Xt")
        yt = self.dm.pull_data_from_device("Yt")
        self.assertEqual(xt[0].mean(), 3)
        self.assertEqual(xt[1].mean(), 0)
        self.assertEqual(xt[2].mean(), 0)
        self.assertAlmostEqual(yt[0].mean(), 0.3, places=6)
        self.assertAlmostEqual(yt[1].mean(), 0.0, places=6)
        self.assertAlmostEqual(yt[2].mean(), 0.0, places=6)
