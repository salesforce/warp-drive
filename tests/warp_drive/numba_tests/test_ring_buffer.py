# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest
import numpy as np
import torch
from warp_drive.managers.numba_managers.numba_data_manager import NumbaDataManager
from warp_drive.training.utils.ring_buffer import RingBuffer, RingBufferManager
from warp_drive.utils.data_feed import DataFeed


class TestRingBuffer(unittest.TestCase):
    """
    Unit tests for the RingBuffer
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = NumbaDataManager(num_agents=5, num_envs=1, episode_length=3)
        self.rbm = RingBufferManager()

    def test(self):
        x = np.zeros((5, 3), dtype=np.float32)
        data = DataFeed()
        data.add_data(name="X", data=x)
        self.dm.push_data_to_device(data, torch_accessible=True)
        self.rbm.add(name="X", data_manager=self.dm)
        buffer = self.rbm.get("X")
        for i in [0, 1, 2]:
            buffer.enqueue(torch.tensor([i, i, i]))

        res1 = buffer.unroll().cpu().numpy()
        self.assertEqual(
            res1.tolist(),
            np.array([[0, 0, 0],
                      [1, 1, 1],
                      [2, 2, 2]]
                     ).tolist()
        )

        self.assertTrue(not buffer.isfull())

        for i in [3, 4]:
            buffer.enqueue(torch.tensor([i, i, i]))
        res2 = buffer.unroll().cpu().numpy()
        self.assertEqual(
            res2.tolist(),
            np.array([[0, 0, 0],
                      [1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [4, 4, 4]]
                     ).tolist()
        )

        self.assertTrue(buffer.isfull())

        for i in [5, 6, 7]:
            buffer.enqueue(torch.tensor([i, i, i]))
        res3 = buffer.unroll().cpu().numpy()
        self.assertEqual(
            res3.tolist(),
            np.array([[3, 3, 3],
                      [4, 4, 4],
                      [5, 5, 5],
                      [6, 6, 6],
                      [7, 7, 7]]
                     ).tolist()
        )

        self.assertTrue(buffer.isfull())







