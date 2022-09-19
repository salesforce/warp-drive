# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from numba import cuda as numba_driver

try:
    from warp_drive.numba_includes.env_config import *
except ImportError:
    raise Exception("warp_drive.numba_includes.env_config is not available")


# we have size of episode_length + 1 timesteps for logging,
# the 0th timestep is reserved for reset to place the data at reset
@numba_driver.jit
def reset_log_mask(log_mask, episode_length):
    tidx = numba_driver.threadIdx.x
    if tidx == 0:
        for i in range(episode_length + 1):
            log_mask[i] = 0


@numba_driver.jit
def update_log_mask(log_mask, timestep, episode_length):
    tidx = numba_driver.threadIdx.x
    if tidx == 0 and timestep <= episode_length:
        if timestep >= 1:
            assert log_mask[timestep - 1] == 1
        log_mask[timestep] = 1


@numba_driver.jit(device=True, inline=True)
def log_one_step_2d_helper(log, data, timestep, agent_id, env_id):
    log[timestep, agent_id] = data[env_id, agent_id]


@numba_driver.jit(device=True, inline=True)
def log_one_step_3d_helper(log, data, feature_dim, timestep, agent_id, env_id):
    for i in range(feature_dim):
        log[timestep, agent_id, i] = data[env_id, agent_id, i]


@numba_driver.jit
def log_one_step_2d(log, data, timestep, episode_length, env_id):
    agent_id = numba_driver.grid(1)
    # only run threads in the first block
    if agent_id >= wkNumberAgents:
        return
    if timestep > episode_length:
        return

    log_one_step_2d_helper(log, data, timestep, agent_id, env_id)


@numba_driver.jit
def log_one_step_3d(log, data, feature_dim, timestep, episode_length, env_id):
    agent_id = numba_driver.grid(1)
    # only run threads in the first block
    if agent_id >= wkNumberAgents:
        return
    if timestep > episode_length:
        return

    log_one_step_3d_helper(log, data, feature_dim, timestep, agent_id, env_id)
