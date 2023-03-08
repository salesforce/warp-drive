// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#include <curand_kernel.h>

#ifndef CUDA_INCLUDES_RANDOM_STATES_H_
#define CUDA_INCLUDES_RANDOM_STATES_H_

__device__ curandState_t* states[wkNumberEnvs * wkNumberAgents];

extern "C" __global__ void init_random(int seed);

extern "C" __global__ void free_random();

// binary search to get the action index
__device__ int search_index(float* distr, float p, int l, int r);

extern "C" __global__ void sample_actions(float*, int*, float*, int, int, int);

#endif  // CUDA_INCLUDES_RANDOM_STATES_H_
