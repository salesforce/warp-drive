// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#ifndef CUDA_INCLUDES_LOG_H_
#define CUDA_INCLUDES_LOG_H_

extern "C" __global__ void reset_log_mask(int* log_mask,
int episode_length);

extern "C" __global__ void update_log_mask(int* log_mask, int timestep,
int episode_length);

extern "C" __global__ void log_one_step_in_float(float* log, float* data,
int feature_dim, int timestep, int episode_length, int env_id = 0);

extern "C" __global__ void log_one_step_in_int(int* log, int* data,
int feature_dim, int timestep, int episode_length, int env_id = 0);

#endif  // CUDA_INCLUDES_LOG_H_


