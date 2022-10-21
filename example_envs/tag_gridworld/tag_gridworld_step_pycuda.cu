// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause
__constant__ int kIndexToActionArr[10];

extern "C" {
  __device__ void CudaTagGridWorldGenerateObservation(
    int *states_x_arr,
    int *states_y_arr,
    float *obs_arr,
    int world_boundary,
    int *env_timestep_arr,
    int episode_length,
    int agent_id,
    int env_id,
    bool use_full_observation
  ) {
    const bool is_tagger = (agent_id < wkNumberAgents - 1);


    // obs shape is (num_envs, num_agents, 4 * num_agents + 1)
    // state shape is (num_envs, num_agents,)
    if (use_full_observation) {
      const int agent_times_feature_dim = wkNumberAgents * 4 + 1;
      const int obs_start_index = env_id * wkNumberAgents * agent_times_feature_dim;
      const int state_index = env_id * wkNumberAgents + agent_id;

      for (int ag_id = 0; ag_id < wkNumberAgents; ag_id++) {
        const int state_x_obs_index =
            obs_start_index + ag_id * agent_times_feature_dim + agent_id;
        const int state_y_obs_index = state_x_obs_index + wkNumberAgents;
        const int type_obs_index = state_y_obs_index + wkNumberAgents;
        const int is_current_agent_obs_index = type_obs_index + wkNumberAgents;

        obs_arr[state_x_obs_index] =
            states_x_arr[state_index] / static_cast<float>(world_boundary);
        obs_arr[state_y_obs_index] =
            states_y_arr[state_index] / static_cast<float>(world_boundary);
        obs_arr[type_obs_index] = 1.0 * static_cast<int>(
          agent_id == wkNumberAgents - 1);
        obs_arr[is_current_agent_obs_index] = 1.0 * static_cast<int>(
          ag_id == agent_id);

        if (agent_id == wkNumberAgents - 1) {
          int time_to_end_index = is_current_agent_obs_index + 1;
          obs_arr[time_to_end_index] =
              env_timestep_arr[env_id] / static_cast<float>(episode_length);
        }
      }
    } else {
      // obs shape is (num_envs, num_agents, 6)
      // state shape is (num_envs, num_agents,)
      __shared__ int distance[wkNumberAgents];

      const int state_index = env_id * wkNumberAgents + agent_id;
      const int obs_start_index = state_index * 6;
      const int adversary_state_index = env_id * wkNumberAgents + wkNumberAgents - 1;

      // tagger and runners observe their own locations
      obs_arr[obs_start_index] =
          states_x_arr[state_index] / static_cast<float>(
            world_boundary);
      obs_arr[obs_start_index + 1] =
          states_y_arr[state_index] / static_cast<float>(
            world_boundary);

      if (is_tagger) {
        // Taggers can observe the runner location
        obs_arr[obs_start_index + 2] =
            states_x_arr[adversary_state_index] / static_cast<float>(
              world_boundary);
        obs_arr[obs_start_index + 3] =
            states_y_arr[adversary_state_index] / static_cast<float>(
              world_boundary);
        distance[agent_id] =
            pow(states_x_arr[state_index] -
              states_x_arr[adversary_state_index], 2) +
            pow(states_y_arr[state_index] -
              states_y_arr[adversary_state_index], 2);
      }

      __syncthreads();


      // A runner can observe the tagger location closest to it.
      if (!is_tagger) {
        int closest_agent_id = 0;
        int min_distance = 2 * world_boundary * world_boundary;
        for (int ag_id = 0; ag_id < wkNumberAgents - 1; ag_id++) {
          if (distance[ag_id] < min_distance) {
            min_distance = distance[ag_id];
            closest_agent_id = ag_id;
          }
        }
        obs_arr[obs_start_index + 2] =
            states_x_arr[env_id * wkNumberAgents + closest_agent_id] /
            static_cast<float>(world_boundary);
        obs_arr[obs_start_index + 3] =
            states_y_arr[env_id * wkNumberAgents + closest_agent_id] /
            static_cast<float>(world_boundary);
      }

      obs_arr[obs_start_index + 4] = 1.0 * static_cast<int>(
          agent_id == wkNumberAgents - 1);
      obs_arr[obs_start_index + 5] =
          env_timestep_arr[env_id] / static_cast<float>(episode_length);
    }
  }

  __global__ void
  CudaTagGridWorldStep(
    int *states_x_arr,
    int *states_y_arr,
    int *actions_arr,
    int *done_arr,
    float *rewards_arr,
    float *obs_arr,
    float wall_hit_penalty,
    float tag_reward_for_tagger,
    float tag_penalty_for_runner,
    float step_cost_for_tagger,
    bool use_full_observation,
    int world_boundary,
    int *env_timestep_arr,
    int episode_length
  ) {
    // This implements Tagger on a discrete grid.
    // There are N taggers and 1 runner.
    // The taggers try to tag the runner.
    __shared__ int num_total_tagged;

    const int kEnvId = blockIdx.x;
    const int kThisAgentId = threadIdx.x;
    const bool is_tagger = (kThisAgentId < wkNumberAgents - 1);


    // Increment time ONCE -- only 1 thread can do this.
    // Initialize the shared variable that counts how many runners are tagged.
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
      num_total_tagged = 0;
    }

    __syncthreads();

    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId]
      <= episode_length);

    int global_state_arr_shape[] = {gridDim.x, wkNumberAgents};
    int agent_index[] = {kEnvId, kThisAgentId};
    int adv_agent_index[] = {kEnvId, wkNumberAgents - 1};
    int dimension = 2;
    int state_index = get_flattened_array_index(agent_index, global_state_arr_shape, dimension);
    int adversary_state_index = get_flattened_array_index(adv_agent_index, global_state_arr_shape, dimension);
    int action_index = get_flattened_array_index(agent_index, global_state_arr_shape, dimension);
    int reward_index = get_flattened_array_index(agent_index, global_state_arr_shape, dimension);

    rewards_arr[reward_index] = 0.0;

    float __rew = 0.0;


    // -----------------------------------
    // Movement
    // -----------------------------------
    // Take action and check boundary cost.
    // Map action index to the real action space.
    int ac_index;
    ac_index = actions_arr[action_index] * 2;

    states_x_arr[state_index] = states_x_arr[state_index] +
    kIndexToActionArr[ac_index];
    states_y_arr[state_index] = states_y_arr[state_index] +
    kIndexToActionArr[ac_index + 1];

    if (states_x_arr[state_index] < 0) {
      states_x_arr[state_index] = 0;
      __rew -= wall_hit_penalty;
    } else if (states_x_arr[state_index] > world_boundary) {
      states_x_arr[state_index] = world_boundary;
      __rew -= wall_hit_penalty;
    }

    if (states_y_arr[state_index] < 0) {
      states_y_arr[state_index] = 0;
      __rew -= wall_hit_penalty;
    } else if (states_y_arr[state_index] > world_boundary) {
      states_y_arr[state_index] = world_boundary;
      __rew -= wall_hit_penalty;
    }

    // make sure all agents have finished their movements
    __syncthreads();

    // -----------------------------------
    // Check tags
    // -----------------------------------
    // If this agent is a tagger, check number of tags
    if (is_tagger) {
      if (states_x_arr[state_index] == states_x_arr[adversary_state_index] &&
          states_y_arr[state_index] == states_y_arr[adversary_state_index]) {
        atomicAdd(&num_total_tagged, 1);
      }
    }

    // make sure all agents have finished tag count
    __syncthreads();


    // -----------------------------------
    // Rewards
    // -----------------------------------
    // If this agent is a tagger.
    if (is_tagger) {
      if (num_total_tagged > 0) {
        __rew += tag_reward_for_tagger;
      } else {
        __rew -= step_cost_for_tagger;
      }
    } else {  // If it's the runner.
      if (num_total_tagged > 0) {
        __rew -= tag_penalty_for_runner;
      } else {
        __rew += step_cost_for_tagger;
      }
    }

    rewards_arr[reward_index] = __rew;


    // -----------------------------------
    // Generate observation.
    // -----------------------------------
    // (x, y, tagger or runner, current_agent_or_not)
    CudaTagGridWorldGenerateObservation(states_x_arr, states_y_arr,
      obs_arr, world_boundary, env_timestep_arr, episode_length,
      kThisAgentId, kEnvId, use_full_observation);


    // -----------------------------------
    // End condition
    // -----------------------------------
    // Determine if we're done (the runner is tagged or not).
    if (env_timestep_arr[kEnvId] == episode_length || num_total_tagged > 0) {
      if (kThisAgentId == 0) {
        done_arr[kEnvId] = 1;
      }
    }
  }
}
