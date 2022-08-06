# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest

import numpy as np

from example_envs.tag_continuous.tag_continuous import TagContinuous
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU

# Env configs for testing
env_configs = {
    "test1": {
        "num_taggers": 2,
        "num_runners": 3,
        "max_acceleration": 1,
        "max_turn": np.pi / 4,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "seed": 274880,
        "skill_level_runner": 1,
        "skill_level_tagger": 1,
        "use_full_observation": True,
        "runner_exits_game_after_tagged": True,
        "tagging_distance": 0.0,
    },
    "test2": {
        "num_taggers": 4,
        "num_runners": 1,
        "max_acceleration": 0.05,
        "max_turn": np.pi / 4,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "step_penalty_for_tagger": -0.1,
        "seed": 428096,
        "skill_level_runner": 1,
        "skill_level_tagger": 2,
        "use_full_observation": False,
        "runner_exits_game_after_tagged": False,
        "tagging_distance": 0.25,
    },
    "test3": {
        "num_taggers": 1,
        "num_runners": 4,
        "max_acceleration": 2,
        "max_turn": np.pi / 2,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "step_reward_for_runner": 0.1,
        "seed": 654208,
        "skill_level_runner": 1,
        "skill_level_tagger": 0.5,
        "use_full_observation": False,
        "runner_exits_game_after_tagged": True,
    },
    "test4": {
        "num_taggers": 3,
        "num_runners": 2,
        "max_acceleration": 0.05,
        "max_turn": np.pi,
        "num_acceleration_levels": 3,
        "num_turn_levels": 3,
        "grid_length": 10,
        "episode_length": 100,
        "seed": 121024,
        "skill_level_runner": 0.5,
        "skill_level_tagger": 1,
        "use_full_observation": True,
        "runner_exits_game_after_tagged": False,
    },
}


class MyTestCase(unittest.TestCase):
    """
    CPU v GPU consistency unit tests
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.testing_class = EnvironmentCPUvsGPU(
            dual_mode_env_class=TagContinuous,
            env_configs=env_configs,
            num_envs=2,
            num_episodes=2,
            use_gpu_testing_mode=True,
        )

    def test_env_consistency(self):
        try:
            self.testing_class.test_env_reset_and_step(seed=274880)
        except AssertionError:
            self.fail("TagContinuous environment consistency tests failed")
