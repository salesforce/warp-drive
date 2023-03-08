// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#include "random.h"

__constant__ float kEps = 1.0e-8;


// init kernel randomness seed
// used as random seed for sampling with given distribution
__global__ void init_random(int seed) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= wkNumberEnvs * wkNumberAgents)
      return;
  curandState_t* s = new curandState_t;
  if (s != 0) {
    curand_init(seed, tidx, 0, s);
  }
  states[tidx] = s;
}

__global__ void free_random() {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= wkNumberEnvs * wkNumberAgents)
      return;
  curandState_t* s = states[tidx];
  delete s;
}

__device__ int search_index(float* distr, float p, int l, int r) {
  int mid;
  int left = l;
  int right = r;

  while (left <= right) {
    mid = left + (right - left) / 2;
    if (abs(distr[mid] - p) < kEps) {
      return mid - l;
    } else if (distr[mid] < p) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return left > r ? r -l : left - l;
}

__global__ void sample_actions(float* distr, int* action_indices,
float* cum_distr, int num_agents, int num_actions, int use_argmax) {
  int posidx = blockIdx.x*blockDim.x + threadIdx.x;
  if (posidx >= wkNumberEnvs * num_agents)
      return;
  int dist_index = posidx * num_actions;

  if (use_argmax > 0.5){
    float max_p = distr[dist_index];
    int max_ind = 0;
    for (int i = 1; i < num_actions; i++){
      if (max_p < distr[dist_index + i]){
        max_p = distr[dist_index + i];
        max_ind = i;
      }
    }
    action_indices[posidx] = max_ind;
    return;
  }

  curandState_t s = *states[posidx];
  float p = curand_uniform(&s);
  *states[posidx] = s;

  cum_distr[dist_index] = distr[dist_index];

  for (int i = 1; i < num_actions; i++) {
    cum_distr[dist_index + i] = distr[dist_index + i] +
    cum_distr[dist_index + i - 1];
  }

  int ind = search_index(cum_distr, p, dist_index,
    dist_index + num_actions-1);
  action_indices[posidx] = ind;
}
