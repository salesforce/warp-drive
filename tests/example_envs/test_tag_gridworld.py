# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import unittest

from example_envs.tag_gridworld.tag_gridworld import CUDATagGridWorld, TagGridWorld
from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU

# Env configs for testing
env_configs = {
    "test1": {
        "num_taggers": 4,
        "grid_length": 4,
        "episode_length": 20,
        "seed": 27,
        "wall_hit_penalty": 0.1,
        "tag_reward_for_tagger": 10.0,
        "tag_penalty_for_runner": 2.0,
        "step_cost_for_tagger": 0.01,
        "use_full_observation": True,
    },
    "test2": {
        "num_taggers": 4,
        "grid_length": 4,
        "episode_length": 20,
        "seed": 27,
        "wall_hit_penalty": 0.1,
        "tag_reward_for_tagger": 10.0,
        "tag_penalty_for_runner": 2.0,
        "step_cost_for_tagger": 0.01,
        "use_full_observation": False,
    },
}


class MyTestCase(unittest.TestCase):
    """
    CPU v GPU consistency unit tests
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.testing_class = EnvironmentCPUvsGPU(
            cpu_env_class=TagGridWorld,
            cuda_env_class=CUDATagGridWorld,
            env_configs=env_configs,
            num_envs=2,
            num_episodes=2,
            use_gpu_testing_mode=True,
        )

    def test_env_consistency(self):
        try:
            self.testing_class.test_env_reset_and_step()
        except AssertionError:
            self.fail("TagGridWorld environment consistency tests failed")
