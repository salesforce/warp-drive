from numba import int32, float32
from numba import cuda as numba_driver


@numba_driver.jit([(int32[:,::1], int32[:,::1], int32, int32, int32),
                   (float32[:,::1], float32[:,::1], int32, int32, int32)])
def reset_when_done_2d(data, ref, done, feature_dim, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if force_reset > 0.5 or done[env_id] > 0.5:
        if tid < feature_dim:
            data[env_id][tid] = ref[env_id][tid]


@numba_driver.jit([(int32[:,:,::1], int32[:,:,::1], int32, int32, int32, int32),
                   (float32[:,:,::1], float32[:,:,::1], int32, int32, int32, int32)])
def reset_when_done_3d(data, ref, done, agent_dim, feature_dim, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if force_reset > 0.5 or done[env_id] > 0.5:
        if tid < agent_dim:
            for i in range(feature_dim):
                data[env_id][tid][i] = ref[env_id][tid][i]