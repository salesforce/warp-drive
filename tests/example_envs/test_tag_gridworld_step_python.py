# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np

from example_envs.tag_gridworld import tag_gridworld


class TestPyEnvTagGridWorld(unittest.TestCase):
    """
    Unit tests for the (Python) step function in Tag GridWorld
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = None

    def test_step_case_1(self):
        # first step, assume got the spread agent distribution
        # and adversary distribution
        # (used 100%, so it is fixed for testing)

        self.env = tag_gridworld.TagGridWorld(
            num_taggers=4,
            grid_length=4,
            episode_length=10,
            starting_location_x=np.array([0, 0, 0, 0, 1]),
            starting_location_y=np.array([0, 0, 0, 0, 1]),
            wall_hit_penalty=0.1,
            tag_reward_for_tagger=10.0,
            tag_penalty_for_runner=2.0,
            step_cost_for_tagger=0.01,
        )

        self.env.reset()
        actions = {0: 2, 1: 0, 2: 3, 3: 4, 4: 2}
        observations, rewards, done, _ = self.env.step(actions)
        observations_update = np.array([obs for _, obs in observations.items()])
        rewards_update = np.array([rew for _, rew in rewards.items()])
        done_update = done["__all__"]

        ref_rewards = np.array([9.9, 10.0, 10.0, 9.9, -2.0])

        ref_observations = np.array(
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
                    0.4,
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
                    0.4,
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
                    0.4,
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
                    0.4,
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
                    0.4,
                ],
            ]
        )

        self.assertTrue(np.absolute(rewards_update - ref_rewards).max() < 1e-5)
        self.assertTrue(
            np.absolute(observations_update.reshape(5, -1) * 4 - ref_observations).max()
            < 1e-5
        )
        self.assertEqual(done_update, True)

    def test_step_case_2(self):
        # first step, assume got the spread agent distribution
        # and adversary distribution
        # (used 100%, so it is fixed for testing)

        self.env = tag_gridworld.TagGridWorld(
            num_taggers=4,
            grid_length=4,
            episode_length=10,
            starting_location_x=np.array([1, 3, 3, 1, 0]),
            starting_location_y=np.array([1, 1, 3, 3, 1]),
            wall_hit_penalty=0.1,
            tag_reward_for_tagger=10.0,
            tag_penalty_for_runner=2.0,
            step_cost_for_tagger=0.01,
        )

        o = self.env.reset()
        print(o)
        actions = {0: 1, 1: 3, 2: 2, 3: 4, 4: 1}
        observations, rewards, done, _ = self.env.step(actions)
        observations_update = np.array([obs for _, obs in observations.items()])
        rewards_update = np.array([rew for _, rew in rewards.items()])
        done_update = done["__all__"]

        ref_rewards = np.array([-0.01, -0.01, -0.01, -0.01, 0.01])

        ref_observations = np.array(
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
                    0.4,
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
                    0.4,
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
                    0.4,
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
                    0.4,
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
                    0.4,
                ],
            ]
        )

        self.assertTrue(np.absolute(rewards_update - ref_rewards).max() < 1e-5)
        self.assertTrue(
            np.absolute(observations_update.reshape(5, -1) * 4 - ref_observations).max()
            < 1e-5
        )
        self.assertEqual(done_update, False)

        actions = {0: 2, 1: 1, 2: 0, 3: 4, 4: 0}
        observations, rewards, done, _ = self.env.step(actions)
        observations_update = np.array([obs for _, obs in observations.items()])
        rewards_update = np.array([rew for _, rew in rewards.items()])
        done_update = done["__all__"]

        ref_rewards = np.array([10.0, 10.0, 10.0, 10.0, -2.0])

        ref_observations = np.array(
            [
                [
                    1.0,
                    4.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
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
                    0.8,
                ],
                [
                    1.0,
                    4.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
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
                    0.8,
                ],
                [
                    1.0,
                    4.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
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
                    0.8,
                ],
                [
                    1.0,
                    4.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
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
                    0.8,
                ],
                [
                    1.0,
                    4.0,
                    2.0,
                    1.0,
                    1.0,
                    1.0,
                    2.0,
                    3.0,
                    1.0,
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
                    0.8,
                ],
            ]
        )

        self.assertTrue(np.absolute(rewards_update - ref_rewards).max() < 1e-5)
        self.assertTrue(
            np.absolute(observations_update.reshape(5, -1) * 4 - ref_observations).max()
            < 1e-5
        )
        self.assertEqual(done_update, True)
