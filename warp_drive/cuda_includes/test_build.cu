// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#ifndef CUDA_INCLUDES_TEST_BUILD_CONST_H_
#define CUDA_INCLUDES_TEST_BUILD_CONST_H_
const int wkNumberEnvs = 2;
const int wkNumberAgents = 5;
const int wkBlocksPerEnv = 1;

#include "./core_service.h"
#include "../../example_envs/dummy_env/test_step.cu"
#include "../../example_envs/tag_gridworld/tag_gridworld_step_pycuda.cu"
#include "../../example_envs/tag_continuous/tag_continuous_step_pycuda.cu"

#endif  // CUDA_INCLUDES_TEST_BUILD_CONST_H_