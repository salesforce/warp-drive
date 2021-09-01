// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#ifndef CUDA_INCLUDES_RESET_H_
#define CUDA_INCLUDES_RESET_H_

extern "C" __global__ void reset_in_float_when_done_2d(float* data,
    const float* ref, int* done, int feature_dim, int force_reset = 0);

extern "C" __global__ void reset_in_int_when_done_2d(int* data,
    const int* ref, int* done, int feature_dim, int force_reset = 0);

extern "C" __global__ void reset_in_float_when_done_3d(float* data,
    const float* ref, int* done, int agent_dim, int feature_dim,
    int force_reset = 0);

extern "C" __global__ void reset_in_int_when_done_3d(int* data,
    const int* ref, int* done, int agent_dim, int feature_dim,
    int force_reset = 0);

extern "C" __global__ void undo_done_flag_and_reset_timestep(
    int* done, int* timestep, int force_reset = 0);

#endif  // CUDA_INCLUDES_RESET_H_