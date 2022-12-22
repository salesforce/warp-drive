# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
import math
import numpy as np
import numba.cuda as numba_driver
from numba import float32, int32, boolean
try:
    from warp_drive.numba_includes.env_config import *
except ImportError:
    raise Exception("warp_drive.numba_includes.env_config is not available")

kIndexToActionArr = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])


@numba_driver.jit((int32[:, ::1],
                   int32[:, ::1],
                   float32[:, :, ::1],
                   int32,
                   int32[::1],
                   int32,
                   int32,
                   int32,
                   boolean),
                  device=True)
def NumbaTagGridWorldGenerateObservation(
    states_x_arr,
    states_y_arr,
    obs_arr,
    world_boundary,
    env_timestep_arr,
    episode_length,
    agent_id,
    env_id,
    use_full_observation
):

    is_tagger = (agent_id < wkNumberAgents - 1)

    if use_full_observation:
        # obs shape is (num_envs, num_agents, 4 * num_agents + 1)
        # state shape is (num_envs, num_agents,)
        for ag_id in range(wkNumberAgents):
            obs_arr[env_id, ag_id, agent_id] = states_x_arr[env_id, agent_id] / float(world_boundary)
            obs_arr[env_id, ag_id, agent_id + wkNumberAgents] = states_y_arr[env_id, agent_id] / float(world_boundary)
            obs_arr[env_id, ag_id, agent_id + 2 * wkNumberAgents] = 1.0 * int(agent_id == wkNumberAgents - 1)
            obs_arr[env_id, ag_id, agent_id + 3 * wkNumberAgents] = 1.0 * int(ag_id == agent_id)
            if agent_id == wkNumberAgents - 1:
                obs_arr[env_id, ag_id, agent_id + 3 * wkNumberAgents + 1] = \
                    env_timestep_arr[env_id] / float(episode_length)
    else:
        # obs shape is (num_envs, num_agents, 6)
        # state shape is (num_envs, num_agents,)
        distance = numba_driver.shared.array(shape=(wkNumberAgents,), dtype=np.int32)
        obs_arr[env_id, agent_id, 0] = states_x_arr[env_id, agent_id] / float(world_boundary)
        obs_arr[env_id, agent_id, 1] = states_y_arr[env_id, agent_id] / float(world_boundary)

        if is_tagger:
            obs_arr[env_id, agent_id, 2] = states_x_arr[env_id, wkNumberAgents - 1] / float(world_boundary)
            obs_arr[env_id, agent_id, 3] = states_y_arr[env_id, wkNumberAgents - 1] / float(world_boundary)
            distance[agent_id] = math.pow(states_x_arr[env_id, agent_id] -
                                          states_x_arr[env_id, wkNumberAgents - 1], 2) + \
                                 math.pow(states_y_arr[env_id, agent_id] -
                                          states_y_arr[env_id, wkNumberAgents - 1], 2)

        numba_driver.syncthreads()

        if not is_tagger:
            closest_agent_id = 0
            min_distance = 2 * world_boundary * world_boundary
            for ag_id in range(wkNumberAgents - 1):
                if distance[ag_id] < min_distance:
                    min_distance = distance[ag_id]
                    closest_agent_id = ag_id
            obs_arr[env_id, agent_id, 2] = states_x_arr[env_id, closest_agent_id] / float(world_boundary)
            obs_arr[env_id, agent_id, 3] = states_y_arr[env_id, closest_agent_id] / float(world_boundary)

        obs_arr[env_id, agent_id, 4] = 1.0 * int(agent_id == wkNumberAgents - 1)
        obs_arr[env_id, agent_id, 5] = env_timestep_arr[env_id] / float(episode_length)


@numba_driver.jit((int32[:, ::1],
                   int32[:, ::1],
                   int32[:, :, ::1],
                   int32[::1],
                   float32[:, ::1],
                   float32[:, :, ::1],
                   float32,
                   float32,
                   float32,
                   float32,
                   boolean,
                   int32,
                   int32[::1],
                   int32))
def NumbaTagGridWorldStep(
    states_x_arr,
    states_y_arr,
    actions_arr,
    done_arr,
    rewards_arr,
    obs_arr,
    wall_hit_penalty,
    tag_reward_for_tagger,
    tag_penalty_for_runner,
    step_cost_for_tagger,
    use_full_observation,
    world_boundary,
    env_timestep_arr,
    episode_length
):
    # This implements Tagger on a discrete grid.
    # There are N taggers and 1 runner.
    # The taggers try to tag the runner.
    kEnvId = numba_driver.blockIdx.x
    kThisAgentId = numba_driver.threadIdx.x
    is_tagger = (kThisAgentId < wkNumberAgents - 1)

    num_total_tagged = numba_driver.shared.array(shape=(1,), dtype=np.int32)

    kAction = numba_driver.const.array_like(kIndexToActionArr)

    # Increment time ONCE -- only 1 thread can do this.
    # Initialize the shared variable that counts how many runners are tagged.
    if kThisAgentId == 0:
        env_timestep_arr[kEnvId] += 1
        num_total_tagged[0] = 0

    numba_driver.syncthreads()

    assert 0 < env_timestep_arr[kEnvId] <= episode_length

    rewards_arr[kEnvId, kThisAgentId] = 0.0

    __rew = 0.0

    # -----------------------------------
    # Movement
    # -----------------------------------
    # Take action and check boundary cost.
    # Map action index to the real action space.
    ac_index = actions_arr[kEnvId, kThisAgentId, 0]

    states_x_arr[kEnvId, kThisAgentId] = states_x_arr[kEnvId, kThisAgentId] + kAction[ac_index, 0]
    states_y_arr[kEnvId, kThisAgentId] = states_y_arr[kEnvId, kThisAgentId] + kAction[ac_index, 1]

    if states_x_arr[kEnvId, kThisAgentId] < 0:
        states_x_arr[kEnvId, kThisAgentId] = 0
        __rew -= wall_hit_penalty
    elif states_x_arr[kEnvId, kThisAgentId] > world_boundary:
        states_x_arr[kEnvId, kThisAgentId] = world_boundary
        __rew -= wall_hit_penalty

    if states_y_arr[kEnvId, kThisAgentId] < 0:
        states_y_arr[kEnvId, kThisAgentId] = 0
        __rew -= wall_hit_penalty
    elif states_y_arr[kEnvId, kThisAgentId] > world_boundary:
        states_y_arr[kEnvId, kThisAgentId] = world_boundary
        __rew -= wall_hit_penalty

    # make sure all agents have finished their movements
    numba_driver.syncthreads()

    # -----------------------------------
    # Check tags
    # -----------------------------------
    if is_tagger:
        if states_x_arr[kEnvId, kThisAgentId] == states_x_arr[kEnvId, wkNumberAgents - 1] and \
                states_y_arr[kEnvId, kThisAgentId] == states_y_arr[kEnvId, wkNumberAgents - 1]:
            numba_driver.atomic.add(num_total_tagged, 0, 1)

    # make sure all agents have finished tag count
    numba_driver.syncthreads()

    # -----------------------------------
    # Rewards
    # -----------------------------------
    if is_tagger:
        if num_total_tagged[0] > 0:
            __rew += tag_reward_for_tagger
        else:
            __rew -= step_cost_for_tagger
    else:
        if num_total_tagged[0] > 0:
            __rew -= tag_penalty_for_runner
        else:
            __rew += step_cost_for_tagger

    rewards_arr[kEnvId, kThisAgentId] = __rew

    # -----------------------------------
    # Generate observation.
    # -----------------------------------
    # (x, y, tagger or runner, current_agent_or_not)
    NumbaTagGridWorldGenerateObservation(states_x_arr,
                                         states_y_arr,
                                         obs_arr,
                                         world_boundary,
                                         env_timestep_arr,
                                         episode_length,
                                         kThisAgentId,
                                         kEnvId,
                                         use_full_observation)

    # -----------------------------------
    # End condition
    # -----------------------------------
    # Determine if we're done (the runner is tagged or not).
    if env_timestep_arr[kEnvId] == episode_length or num_total_tagged[0] > 0:
        if kThisAgentId == 0:
            done_arr[kEnvId] = 1
