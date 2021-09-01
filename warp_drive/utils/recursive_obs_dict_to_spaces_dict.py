# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from gym import spaces

BIG_NUMBER = 1e20


def recursive_obs_dict_to_spaces_dict(obs):
    """Recursively return the observation space dictionary
    for a dictionary of observations

    Args:
        obs (dict): A dictionary of observations keyed by agent index
        for a multi-agent environment

    Returns:
        spaces.Dict: A dictionary of observation spaces
    """
    assert isinstance(obs, dict)
    dict_of_spaces = {}
    for k, v in obs.items():

        # list of lists are listified np arrays
        _v = v
        if isinstance(v, list):
            _v = np.array(v)
        elif isinstance(v, (int, np.integer, float, np.floating)):
            _v = np.array([v])

        # assign Space
        if isinstance(_v, np.ndarray):
            x = float(BIG_NUMBER)
            box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                x = x // 2
                box = spaces.Box(low=-x, high=x, shape=_v.shape, dtype=_v.dtype)
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[k] = box

        elif isinstance(_v, dict):
            dict_of_spaces[k] = recursive_obs_dict_to_spaces_dict(_v)
        else:
            raise TypeError
    return spaces.Dict(dict_of_spaces)
