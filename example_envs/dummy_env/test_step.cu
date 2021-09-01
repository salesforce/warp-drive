// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause

extern "C"{
// test data transfer between CPU and GPU, and the updates in place
  __global__ void testkernel(float* x, int* y, int* done, int* actions, float multiplier, int target, int step, int episode_length)
  {
    __shared__ int reach_target;
    int env_id = blockIdx.x;
    int agent_id = threadIdx.x;
    int action_dim = 3;

    if(agent_id==0){
      reach_target = 0;
    }

    if(agent_id < num_agents){
        int index = env_id * num_agents + agent_id;
        int action_index = index * action_dim;

        x[index] = x[index] / multiplier;
        y[index] = y[index] * multiplier;

        if(y[index] >= target){
          atomicAdd(&reach_target, 1);
        }

        for(int i=0; i<action_dim; i++){
          actions[action_index + i] = i;
        }

        __syncthreads();

        if(step == episode_length || reach_target > 0){
          if(agent_id == 0){
            done[env_id] = 1;
          }
        }
    }
  }
}