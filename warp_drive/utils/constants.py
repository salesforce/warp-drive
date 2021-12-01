# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

# String names used for observations, actions, rewards and done flags
# Single source of truth


class Constants:
    """
    Constants for WarpDrive
    """

    OBSERVATIONS = "observations"
    ACTIONS = "sampled_actions"
    REWARDS = "rewards"
    DONE_FLAGS = "done_flags"
    PROCESSED_OBSERVATIONS = "processed_observations"
    ACTION_MASK = "action_mask"
