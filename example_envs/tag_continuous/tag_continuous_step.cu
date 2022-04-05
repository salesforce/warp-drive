// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

__constant__ float kPi = 3.141592654;
__constant__ float kTwoPi = 6.283185308;
__constant__ float kEpsilon = 1.0e-10;  // to prevent indeterminate cases

extern "C" {
  // Device helper function to compute distances between two agents
  __device__ float ComputeDistance(
    float * loc_x_arr,
    float * loc_y_arr,
    const int kThisAgentId1,
    const int kThisAgentId2,
    const int kEnvId,
    int kNumAgents
  ) {
    const int index1 = kEnvId * kNumAgents + kThisAgentId1;
    const int index2 = kEnvId * kNumAgents + kThisAgentId2;
    return sqrt(
      pow(loc_x_arr[index1] - loc_x_arr[index2], 2) +
      pow(loc_y_arr[index1] - loc_y_arr[index2], 2));
  }

  // Device helper function to generate observation
  __device__ void CudaTagContinuousGenerateObservation(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    int * agent_types_arr,
    const float kGridLength,
    const float kMaxSpeed,
    const int kNumOtherAgentsObserved,
    int * still_in_the_game_arr,
    const bool kUseFullObservation,
    float * obs_arr,
    float * neighbor_distances_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    int * env_timestep_arr,
    const int kNumAgents,
    const int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    int num_features = 7;

    if (kThisAgentId < kNumAgents) {
      if (kUseFullObservation) {
        // obs shape is (num_envs, kNumAgents,
        // num_features * (kNumAgents - 1) + 1)
        const int kThisAgentIdxOffset = kEnvId * kNumAgents *
          (num_features * (kNumAgents - 1) + 1) +
          kThisAgentId * (num_features * (kNumAgents - 1) + 1);
        // Initialize obs
        int index = 0;
        for (int other_agent_id = 0; other_agent_id < kNumAgents;
        other_agent_id++) {
          if (other_agent_id != kThisAgentId) {
            obs_arr[kThisAgentIdxOffset + 0 * (kNumAgents - 1) + index]
              = 0.0;
            obs_arr[kThisAgentIdxOffset + 1 * (kNumAgents - 1) + index]
              = 0.0;
            obs_arr[kThisAgentIdxOffset + 2 * (kNumAgents - 1) + index]
              = 0.0;
            obs_arr[kThisAgentIdxOffset + 3 * (kNumAgents - 1) + index]
              = 0.0;
            obs_arr[kThisAgentIdxOffset + 4 * (kNumAgents - 1) + index]
              = 0.0;
            obs_arr[kThisAgentIdxOffset + 5 * (kNumAgents - 1) + index]
              = agent_types_arr[other_agent_id];
            obs_arr[kThisAgentIdxOffset + 6 * (kNumAgents - 1) + index]
              = still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id];
            index += 1;
          }
        }
        obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1)] = 0.0;

        // Update obs for agents still in the game
        if (still_in_the_game_arr[kThisAgentArrayIdx]) {
          int index = 0;
          for (int other_agent_id = 0; other_agent_id < kNumAgents;
          other_agent_id++) {
            if (other_agent_id != kThisAgentId) {
              const int kOtherAgentArrayIdx = kEnvId * kNumAgents +
                other_agent_id;
              obs_arr[kThisAgentIdxOffset + 0 * (kNumAgents - 1) + index] =
                static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] -
                loc_x_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
              obs_arr[kThisAgentIdxOffset + 1 * (kNumAgents - 1) + index] =
                static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] -
                loc_y_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
              obs_arr[kThisAgentIdxOffset + 2 * (kNumAgents - 1) + index] =
                static_cast<float>(speed_arr[kOtherAgentArrayIdx] -
                speed_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
              obs_arr[kThisAgentIdxOffset + 3 * (kNumAgents - 1) + index] =
                static_cast<float>(acceleration_arr[kOtherAgentArrayIdx] -
                acceleration_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
              obs_arr[kThisAgentIdxOffset + 4 * (kNumAgents - 1) + index] =
                static_cast<float>(direction_arr[kOtherAgentArrayIdx] -
                direction_arr[kThisAgentArrayIdx]) / (kTwoPi);
              index += 1;
            }
          }
          obs_arr[kThisAgentIdxOffset + num_features * (kNumAgents - 1)] =
            static_cast<float>(env_timestep_arr[kEnvId]) / kEpisodeLength;
        }
      } else {
        // Initialize obs to all zeros
        // obs shape is (num_envs, kNumAgents,
        //   num_features * kNumOtherAgentsObserved + 1)
        const int kThisAgentIdxOffset = kEnvId * kNumAgents *
          (num_features * kNumOtherAgentsObserved + 1) +
          kThisAgentId * (num_features * kNumOtherAgentsObserved + 1);
        for (int idx = 0; idx < kNumOtherAgentsObserved; idx++) {
          obs_arr[kThisAgentIdxOffset + 0 * kNumOtherAgentsObserved + idx] =
            0.0;
          obs_arr[kThisAgentIdxOffset + 1 * kNumOtherAgentsObserved + idx] =
            0.0;
          obs_arr[kThisAgentIdxOffset + 2 * kNumOtherAgentsObserved + idx] =
            0.0;
          obs_arr[kThisAgentIdxOffset + 3 * kNumOtherAgentsObserved + idx] =
            0.0;
          obs_arr[kThisAgentIdxOffset + 4 * kNumOtherAgentsObserved + idx] =
            0.0;
          obs_arr[kThisAgentIdxOffset + 5 * kNumOtherAgentsObserved + idx] =
            0.0;
          obs_arr[kThisAgentIdxOffset + 6 * kNumOtherAgentsObserved + idx] =
            0.0;
        }
        obs_arr[kThisAgentIdxOffset + num_features * kNumOtherAgentsObserved] =
            0.0;

        // Update obs for agents still in the game
        if (still_in_the_game_arr[kThisAgentArrayIdx]) {
          int distance_arr_idx;
          int i_index;
          int j_index;
          int neighbor_ids_sorted_by_distance_arr_idx;

          // Find the nearest agents
          const int kThisAgentArrayIdxOffset = kEnvId * kNumAgents *
            (kNumAgents - 1) + kThisAgentId * (kNumAgents - 1);

          // Initialize neighbor_ids_sorted_by_distance_arr
          // other agents that are still in the same
          int num_valid_other_agents = 0;
          for (int other_agent_id = 0; other_agent_id < kNumAgents;
            other_agent_id++) {
            if ((other_agent_id != kThisAgentId) &&
                (still_in_the_game_arr[kEnvId * kNumAgents + other_agent_id])) {
              neighbor_ids_sorted_by_distance_arr_idx =
                kThisAgentArrayIdxOffset + num_valid_other_agents;
              neighbor_ids_sorted_by_distance_arr[
                neighbor_ids_sorted_by_distance_arr_idx] = other_agent_id;
              num_valid_other_agents++;
            }
          }

          // First, find distances to all the valid agents
          for (int idx = 0; idx < num_valid_other_agents; idx++) {
            distance_arr_idx = kThisAgentArrayIdxOffset + idx;
            neighbor_distances_arr[distance_arr_idx] = ComputeDistance(
              loc_x_arr,
              loc_y_arr,
              kThisAgentId,
              neighbor_ids_sorted_by_distance_arr[distance_arr_idx],
              kEnvId,
              kNumAgents);
          }

          // Find the nearest neighbor agent indices
          for (int i = 0; i < min(num_valid_other_agents,
            kNumOtherAgentsObserved); i++) {
            i_index = kThisAgentArrayIdxOffset + i;

            for (int j = i + 1; j < num_valid_other_agents; j++) {
              j_index = kThisAgentArrayIdxOffset + j;

              if (neighbor_distances_arr[j_index] <
                neighbor_distances_arr[i_index]) {
                float tmp1 = neighbor_distances_arr[i_index];
                neighbor_distances_arr[i_index] =
                    neighbor_distances_arr[j_index];
                neighbor_distances_arr[j_index] = tmp1;

                int tmp2 = neighbor_ids_sorted_by_distance_arr[i_index];
                neighbor_ids_sorted_by_distance_arr[i_index] =
                  neighbor_ids_sorted_by_distance_arr[j_index];
                neighbor_ids_sorted_by_distance_arr[j_index] = tmp2;
              }
            }
          }

          // Save nearest neighbor ids.
          for (int idx = 0; idx < min(num_valid_other_agents,
            kNumOtherAgentsObserved); idx++) {
            const int kNearestNeighborsIdx =
              kEnvId * kNumAgents * kNumOtherAgentsObserved +
              kThisAgentId * kNumOtherAgentsObserved +
              idx;
            nearest_neighbor_ids[kNearestNeighborsIdx] =
              neighbor_ids_sorted_by_distance_arr[
                kThisAgentArrayIdxOffset + idx];
          }

          // Update observation
          for (int idx = 0; idx < min(num_valid_other_agents,
            kNumOtherAgentsObserved); idx++) {
            const int kNearestNeighborsIdx =
              kEnvId * kNumAgents * kNumOtherAgentsObserved +
              kThisAgentId * kNumOtherAgentsObserved +
              idx;
            const int kOtherAgentId = nearest_neighbor_ids[
              kNearestNeighborsIdx];
            const int kOtherAgentArrayIdx = kEnvId * kNumAgents +
                kOtherAgentId;

            const int kThisAgentIdxOffset =
              kEnvId * kNumAgents * (
              num_features * kNumOtherAgentsObserved + 1) +
              kThisAgentId * (num_features * kNumOtherAgentsObserved + 1);

            obs_arr[kThisAgentIdxOffset + 0 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_x_arr[kOtherAgentArrayIdx] -
              loc_x_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
            obs_arr[kThisAgentIdxOffset + 1 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(loc_y_arr[kOtherAgentArrayIdx] -
              loc_y_arr[kThisAgentArrayIdx]) / (sqrt(2.0) * kGridLength);
            obs_arr[kThisAgentIdxOffset + 2 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(speed_arr[kOtherAgentArrayIdx] -
              speed_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 3 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(acceleration_arr[kOtherAgentArrayIdx] -
              acceleration_arr[kThisAgentArrayIdx]) / (kMaxSpeed + kEpsilon);
            obs_arr[kThisAgentIdxOffset + 4 * kNumOtherAgentsObserved + idx] =
              static_cast<float>(direction_arr[kOtherAgentArrayIdx] -
              direction_arr[kThisAgentArrayIdx]) / (kTwoPi);
            obs_arr[kThisAgentIdxOffset + 5 * kNumOtherAgentsObserved + idx] =
              agent_types_arr[kOtherAgentId];
            obs_arr[kThisAgentIdxOffset + 6 * kNumOtherAgentsObserved + idx] =
              still_in_the_game_arr[kOtherAgentArrayIdx];
          }
          obs_arr[
            kThisAgentIdxOffset + num_features * kNumOtherAgentsObserved] =
            static_cast<float>(env_timestep_arr[kEnvId]) / kEpisodeLength;
        }
      }
    }
  }

  // Device helper function to compute rewards
  __device__ void CudaTagContinuousComputeReward(
    float * rewards_arr,
    float * loc_x_arr,
    float * loc_y_arr,
    const float kGridLength,
    float * edge_hit_reward_penalty,
    float * step_rewards_arr,
    int * num_runners_arr,
    int * agent_types_arr,
    const float kDistanceMarginForReward,
    const float kTagRewardForTagger,
    const float kTagPenaltyForRunner,
    const float kEndOfGameRewardForRunner,
    bool kRunnerExitsGameAfterTagged,
    int * still_in_the_game_arr,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength,
    const int kEnvId,
    const int kThisAgentId,
    const int kThisAgentArrayIdx
  ) {
    if (kThisAgentId < kNumAgents) {
      // initialize rewards
      rewards_arr[kThisAgentArrayIdx] = 0.0;

      if (still_in_the_game_arr[kThisAgentArrayIdx]) {
        // Add the edge hit penalty and the  step rewards / penalties
        rewards_arr[kThisAgentArrayIdx] += edge_hit_reward_penalty[
          kThisAgentArrayIdx];
        rewards_arr[kThisAgentArrayIdx] += step_rewards_arr[kThisAgentId];
      }

      // Ensure that all the agents rewards are initialized before we proceed.
      // The rewards are only set by the runners, so this pause is necessary.
      __sync_env_threads();
      float min_dist = kGridLength * sqrt(2.0);
      bool is_runner = !agent_types_arr[kThisAgentId];

      if (is_runner && still_in_the_game_arr[kThisAgentArrayIdx]) {
        int nearest_tagger_id;

        for (int other_agent_id = 0; other_agent_id < kNumAgents;
          other_agent_id++) {
          bool is_tagger = (agent_types_arr[other_agent_id] == 1);

          if (is_tagger) {
            const float dist = ComputeDistance(
              loc_x_arr,
              loc_y_arr,
              kThisAgentId,
              other_agent_id,
              kEnvId,
              kNumAgents);
            if (dist < min_dist) {
              min_dist = dist;
              nearest_tagger_id = other_agent_id;
            }
          }
        }

        if (min_dist < kDistanceMarginForReward) {
          // the runner is tagged.
          rewards_arr[kThisAgentArrayIdx] += kTagPenaltyForRunner;
          rewards_arr[kEnvId * kNumAgents + nearest_tagger_id] +=
            kTagRewardForTagger;

          if (kRunnerExitsGameAfterTagged) {
            still_in_the_game_arr[kThisAgentArrayIdx] = 0;
            num_runners_arr[kEnvId] -= 1;
          }
        }

        // Add end of game reward for runners at the end of the episode.
        if (env_timestep_arr[kEnvId] == kEpisodeLength) {
          rewards_arr[kThisAgentArrayIdx] += kEndOfGameRewardForRunner;
        }
      }

      // Wait here to update the number of runners before determining done_arr
      __sync_env_threads();
      // Use only agent 0's thread to set done_arr
      if (kThisAgentId == 0) {
        if ((env_timestep_arr[kEnvId] == kEpisodeLength) ||
          (num_runners_arr[kEnvId] == 0)) {
            done_arr[kEnvId] = 1;
        }
      }
    }
  }

  __global__ void CudaTagContinuousStep(
    float * loc_x_arr,
    float * loc_y_arr,
    float * speed_arr,
    float * direction_arr,
    float * acceleration_arr,
    int * agent_types_arr,
    float * edge_hit_reward_penalty,
    const float kEdgeHitPenalty,
    const float kGridLength,
    float * acceleration_actions_arr,
    float * turn_actions_arr,
    const float kMaxSpeed,
    const int kNumOtherAgentsObserved,
    float * skill_levels_arr,
    const bool kRunnerExitsGameAfterTagged,
    int * still_in_the_game_arr,
    const bool kUseFullObservation,
    float * obs_arr,
    int * action_indices_arr,
    float * neighbor_distances_arr,
    int * neighbor_ids_sorted_by_distance_arr,
    int * nearest_neighbor_ids,
    float * rewards_arr,
    float * step_rewards_arr,
    int * num_runners_arr,
    const float kDistanceMarginForReward,
    const float kTagRewardForTagger,
    const float kTagPenaltyForRunner,
    const float kEndOfGameRewardForRunner,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength
  ) {
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
    const int kNumActions = 2;

    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
    }

    // Wait here until timestep has been updated
    __sync_env_threads();

    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <=
      kEpisodeLength);

    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = kEnvId * kNumAgents * kNumActions +
        kThisAgentId * kNumActions;
      float delta_acceleration = acceleration_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 0]];
      float delta_turn = turn_actions_arr[action_indices_arr[
        kThisAgentActionIdxOffset + 1]];

      acceleration_arr[kThisAgentArrayIdx] += delta_acceleration;

      direction_arr[kThisAgentArrayIdx] = fmod(
        direction_arr[kThisAgentArrayIdx] + delta_turn, kTwoPi) *
        still_in_the_game_arr[kThisAgentArrayIdx];
      if (direction_arr[kThisAgentArrayIdx] < 0) {
        direction_arr[kThisAgentArrayIdx] = kTwoPi + direction_arr[
          kThisAgentArrayIdx];
      }

      // Speed clipping
      speed_arr[kThisAgentArrayIdx] = min(
          kMaxSpeed * skill_levels_arr[kThisAgentId],
          max(
            0.0,
            speed_arr[kThisAgentArrayIdx] + acceleration_arr[
              kThisAgentArrayIdx])) * still_in_the_game_arr[kThisAgentArrayIdx];

      // Reset acceleration to 0 when speed becomes 0 or
      // kMaxSpeed (multiplied by skill levels)
      if ((speed_arr[kThisAgentArrayIdx] <= 0.0) ||
        (speed_arr[kThisAgentArrayIdx] >=
        (kMaxSpeed * skill_levels_arr[kThisAgentId]))) {
          acceleration_arr[kThisAgentArrayIdx] = 0.0;
      }

      loc_x_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] *
        cos(direction_arr[kThisAgentArrayIdx]);
      loc_y_arr[kThisAgentArrayIdx] += speed_arr[kThisAgentArrayIdx] *
        sin(direction_arr[kThisAgentArrayIdx]);

      // Crossing the edge
      bool has_crossed_edge = (
        (loc_x_arr[kThisAgentArrayIdx] < 0) |
        (loc_x_arr[kThisAgentArrayIdx] > kGridLength) |
        (loc_y_arr[kThisAgentArrayIdx] < 0) |
        (loc_y_arr[kThisAgentArrayIdx] > kGridLength));

      // Clip x and y if agent has crossed edge
      if (has_crossed_edge) {
        if (loc_x_arr[kThisAgentArrayIdx] < 0) {
          loc_x_arr[kThisAgentArrayIdx] = 0.0;
        } else if (loc_x_arr[kThisAgentArrayIdx] > kGridLength) {
          loc_x_arr[kThisAgentArrayIdx] = kGridLength;
        }
        if (loc_y_arr[kThisAgentArrayIdx] < 0) {
          loc_y_arr[kThisAgentArrayIdx] = 0.0;
        } else if (loc_y_arr[kThisAgentArrayIdx] > kGridLength) {
          loc_y_arr[kThisAgentArrayIdx] = kGridLength;
        }

        edge_hit_reward_penalty[kThisAgentArrayIdx] = kEdgeHitPenalty;
      } else {
        edge_hit_reward_penalty[kThisAgentArrayIdx] = 0.0;
      }
    }

    // Make sure all agents have updated their states
    __sync_env_threads();
    // -------------------------------
    // Generate observation
    // -------------------------------
    CudaTagContinuousGenerateObservation(
      loc_x_arr,
      loc_y_arr,
      speed_arr,
      direction_arr,
      acceleration_arr,
      agent_types_arr,
      kGridLength,
      kMaxSpeed,
      kNumOtherAgentsObserved,
      still_in_the_game_arr,
      kUseFullObservation,
      obs_arr,
      neighbor_distances_arr,
      neighbor_ids_sorted_by_distance_arr,
      nearest_neighbor_ids,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx);

    // -------------------------------
    // Compute reward
    // -------------------------------
    CudaTagContinuousComputeReward(
      rewards_arr,
      loc_x_arr,
      loc_y_arr,
      kGridLength,
      edge_hit_reward_penalty,
      step_rewards_arr,
      num_runners_arr,
      agent_types_arr,
      kDistanceMarginForReward,
      kTagRewardForTagger,
      kTagPenaltyForRunner,
      kEndOfGameRewardForRunner,
      kRunnerExitsGameAfterTagged,
      still_in_the_game_arr,
      done_arr,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx);
  }
}
