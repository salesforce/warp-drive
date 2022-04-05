#ifndef CUDA_INCLUDES_ENV_SYNC_H_
#define CUDA_INCLUDES_ENV_SYNC_H_

const int wkMaxBlocks = wkNumberEnvs * wkBlocksPerEnv;
__device__ unsigned char env_sync_block_indicator[wkMaxBlocks];

__device__ int getRankInEnv(const int blockIdx){
    return blockIdx % wkBlocksPerEnv;
}

__device__ void __sync_env_threads();

#endif // CUDA_INCLUDES_ENV_SYNC_H_