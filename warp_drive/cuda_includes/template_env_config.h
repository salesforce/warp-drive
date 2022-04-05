// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

#include <assert.h>

#ifndef CUDA_INCLUDES_TEMPLATE_ENV_CONFIG_H_
#define CUDA_INCLUDES_TEMPLATE_ENV_CONFIG_H_
/*
The followings are three fundamental global constants for WarpDrive simulation and the template
will be populated from Python function_manager API based on the user configuration
`wk` represents WarpDrive Constant
`NumberEnvs`: number of environments
`NumberAgents`: number of agents per environment
`BlocksPerEnv`: number of blocks covering one environment, default is 1
*/
const int wkNumberEnvs = <<N_ENVS>>;
const int wkNumberAgents = <<N_AGENTS>>;
const int wkBlocksPerEnv = <<N_BLOCKS_PER_ENV>>;

#endif  // CUDA_INCLUDES_TEMPLATE_ENV_CONFIG_H_
