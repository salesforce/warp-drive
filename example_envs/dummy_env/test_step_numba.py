# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

# test data transfer between CPU and GPU, and the updates in place
from numba import cuda as numba_driver
from numba import int32

try:
    from warp_drive.numba_includes.env_config import *
except ImportError:
    raise Exception("warp_drive.numba_includes.env_config is not available")


@numba_driver.jit
def testkernel(x, y, done, actions, multiplier, target, step, episode_length):
    reach_target = numba_driver.shared.array(1, dtype=int32)
    env_id = numba_driver.blockIdx.x
    agent_id = numba_driver.threadIdx.x
    # this serves as the leading agent for each block
    # since we have shared memory residing in each block
    action_dim = 3

    if agent_id == 0:
        reach_target[0] = 0

    if agent_id < wkNumberAgents:
        x[env_id, agent_id] = x[env_id, agent_id] / multiplier
        y[env_id, agent_id] = y[env_id, agent_id] * multiplier

        if y[env_id, agent_id] >= target:
            numba_driver.atomic.add(reach_target, 0, 1)

        for i in range(action_dim):
            actions[env_id, agent_id, i] = i

        numba_driver.syncthreads()

        if step == episode_length or reach_target[0] > 0:
            if agent_id == 0:
                numba_driver.atomic.max(done, env_id, 1)
