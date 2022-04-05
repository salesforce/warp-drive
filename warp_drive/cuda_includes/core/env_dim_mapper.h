#ifndef CUDA_INCLUDES_ENV_DIM_MAPPER_H_
#define CUDA_INCLUDES_ENV_DIM_MAPPER_H_

// the original CUDA has threadIdx relevant to the block, and blockIdx relevant to the grid
// helper function to map thread_id to the local agent_id from gpu block index
/*
For an agent, we need both `kThisAgentId` and `kEnvId` together to decide who it is

- wkBlocksPerEnv = 1, meaning each env covers 1 block, this is the default setting and ideal for most scenarios
  in this case, kThisAgentId and kEnvId has the direct correspondence to threadIdx and blockIdx, respectively

- wkBlocksPerEnv > 1, meaning each env covers more than one blocks
for example:
wkBlocksPerEnv = 3, meaning each env covers 3 blocks
blockDim = 10, meaning each block owns 10 threads
then: blockIdx = 2, threadIdx = 6 ==> kThisAgentId = 6 + (2 % 3) * 10 = 26, kEnvId = 2 / 3 = 0
      blockIdx = 5, threadIdx = 6 ==> kThisAgentId = 6 + (5 % 3) * 10 = 26, kEnvId = 5 / 3 = 1
      blockIdx = 5, threadIdx = 3 ==> kThisAgentId = 3 + (5 % 3) * 10 = 23, kEnvId = 5 / 3 = 1
      blockIdx = 4, threadIdx = 9 ==> kThisAgentId = 9 + (4 % 3) * 10 = 19, kEnvId = 4 / 3 = 1
      blockIdx = 3, threadIdx = 7 ==> kThisAgentId = 7 + (3 % 3) * 10 = 7, kEnvId = 3 / 3 = 1
*/
__device__ int getAgentID(const int threadIdx, const int blockIdx, const int blockDim){
    int kThisAgentId = (wkBlocksPerEnv > 1) ? threadIdx + (blockIdx % wkBlocksPerEnv) * blockDim : threadIdx;
    return kThisAgentId;
}


__device__ int getEnvID(const int blockIdx){
    int kEnvId = blockIdx / wkBlocksPerEnv;
    return kEnvId;
}

#endif // CUDA_INCLUDES_ENV_DIM_MAPPER_H_

