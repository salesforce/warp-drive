// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#include "reset.h"

__global__ void reset_in_float_when_done_2d(float* data, const float* ref,
  int* done, int feature_dim, int force_reset) {
  int env_id = getEnvID(blockIdx.x);
  int tid = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
  if (force_reset > 0.5 || done[env_id] > 0.5) {
    if (tid < feature_dim) {
      int data_index = env_id * feature_dim + tid;
      data[data_index] = ref[data_index];
    }
  }
}

__global__ void reset_in_int_when_done_2d(int* data, const int* ref,
int* done, int feature_dim, int force_reset) {
  int env_id = getEnvID(blockIdx.x);
  int tid = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
  if (force_reset > 0.5 || done[env_id] > 0.5) {
    if (tid < feature_dim) {
      int data_index = env_id * feature_dim + tid;
      data[data_index] = ref[data_index];
    }
  }
}

template <class T>
__device__ void reset_helper_3d(T* data, const T* ref, int feature_dim,
int data_index) {
  for (int i = 0; i < feature_dim; i++) {
    data[data_index + i] = ref[data_index + i];
  }
}

__global__ void reset_in_float_when_done_3d(float* data, const float* ref,
int* done, int agent_dim, int feature_dim, int force_reset) {
  int env_id = getEnvID(blockIdx.x);
  int tid = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
  if (force_reset > 0.5 || done[env_id] > 0.5) {
    if (tid < agent_dim) {
      int data_index = env_id * agent_dim * feature_dim + tid * feature_dim;
      reset_helper_3d(data, ref, feature_dim, data_index);
    }
  }
}

__global__ void reset_in_int_when_done_3d(int* data, const int* ref,
int* done, int agent_dim, int feature_dim, int force_reset) {
  int env_id = getEnvID(blockIdx.x);
  int tid = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
  if (force_reset > 0.5 || done[env_id] > 0.5) {
    if (tid < agent_dim) {
      int data_index = env_id * agent_dim * feature_dim + tid * feature_dim;
      reset_helper_3d(data, ref, feature_dim, data_index);
    }
  }
}

__global__ void undo_done_flag_and_reset_timestep(int* done, int* timestep,
int force_reset) {
  int agent_id = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
  int env_id = getEnvID(blockIdx.x);
  if (force_reset > 0.5 || done[env_id] > 0.5) {
    if (agent_id == 0) {
      done[env_id] = 0;
      timestep[env_id] = 0;
    }
  }
}
