#include "env_thread_sync.h"


__device__ unsigned char load_arrived(unsigned char *arrived) {
#if __CUDA_ARCH__ < 700
    return *(volatile unsigned char *)arrived;
#else
    unsigned int result;
    asm volatile("ld.acquire.gpu.global.u8 %0, [%1];"
                 : "=r"(result)
                 : "l"(arrived)
                 : "memory");
    return result;
#endif
  }

__device__ void store_arrived(unsigned char *arrived, unsigned char val) {
#if __CUDA_ARCH__ < 700
    *(volatile unsigned char *)arrived = val;
#else
    unsigned int reg_val = val;
    asm volatile(
        "st.release.gpu.global.u8 [%1], %0;" ::"r"(reg_val) "l"(arrived)
        : "memory");

    // Avoids compiler warnings from unused variable val.
    (void)(reg_val = reg_val);
#endif
  }

__device__ void __sync_env_threads(){
    __syncthreads();

    if(wkBlocksPerEnv <= 1) return;

    const int kThisEnvId = getEnvID(blockIdx.x);
    const int blockInEnv = getRankInEnv(blockIdx.x);
    unsigned char* const local_indicator = env_sync_block_indicator + kThisEnvId * wkBlocksPerEnv;

    if(threadIdx.x == 0){
        if(blockInEnv == 0){
            // Leader block waits for others to join and then releases them.
            // Other blocks in env can arrive in any order, so the leader have to wait for
            // all others.
            for (int i = 0; i < wkBlocksPerEnv - 1; i++) {
                while (load_arrived(&local_indicator[i]) == 0)
                    ;
            }
            for (int i = 0; i < wkBlocksPerEnv - 1; i++) {
                store_arrived(&local_indicator[i], 0);
            }
            __threadfence();
        }else{
            // Other blocks in env note their arrival and wait to be released.
            store_arrived(&local_indicator[blockInEnv - 1], 1);
            while (load_arrived(&local_indicator[blockInEnv - 1]) == 1)
                ;
        }
    }

    __syncthreads();
}
