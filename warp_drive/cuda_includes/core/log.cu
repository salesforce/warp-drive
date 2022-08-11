// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#include "log.h"

// we have size of episode_length + 1 timesteps for logging,
// the 0th timestep is reserved for reset to place the data at reset
__global__ void reset_log_mask(int* log_mask, int episode_length) {
  int tidx = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
  if (tidx == 0) {
    for (int i = 0; i<= episode_length; i++) {
      log_mask[i] = 0;
    }
  }
}

__global__ void update_log_mask(int* log_mask, int timestep,
  int episode_length) {
  int tidx = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
  if (tidx == 0 && timestep <= episode_length) {
      if (timestep >= 1) {
        assert(log_mask[timestep-1] == 1);
      }
      log_mask[timestep] = 1;
  }
}

template <class T>
__device__ void log_one_step_helper(T* log, T* data, int feature_dim,
int timestep, int agent_id, int env_id) {
  int time_dependent_log_index = timestep * wkNumberAgents * feature_dim +
    agent_id * feature_dim;
  int data_index = env_id * wkNumberAgents * feature_dim + agent_id * feature_dim;

  for (int i = 0; i < feature_dim; i++) {
    log[time_dependent_log_index + i] = data[data_index + i];
  }
}

__global__ void log_one_step_in_float(float* log, float* data, int feature_dim,
int timestep, int episode_length, int env_id) {
  int agent_id = threadIdx.x + blockIdx.x * blockDim.x;;
  // only run threads in the first block
  if (agent_id >= wkNumberAgents) return;
  if (timestep > episode_length) return;

  log_one_step_helper(log, data, feature_dim, timestep, agent_id, env_id);
}

__global__ void log_one_step_in_int(int* log, int* data, int feature_dim,
int timestep, int episode_length, int env_id) {
  int agent_id = threadIdx.x + blockIdx.x * blockDim.x;

  // only run threads in the first block
  if (agent_id >= wkNumberAgents) return;
  if (timestep > episode_length) return;

  log_one_step_helper(log, data, feature_dim, timestep, agent_id, env_id);
}
