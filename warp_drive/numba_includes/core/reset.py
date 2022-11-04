from numba import cuda as numba_driver
from numba import float32, int32


# @numba_driver.jit([(int32[:], int32[:], int32[:], int32),
#                    (float32[:], float32[:], int32[:], int32)])
@numba_driver.jit
def reset_when_done_1d(data, ref, done, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if tid == 0:
        if force_reset > 0.5 or done[env_id] > 0.5:
            data[env_id] = ref[env_id]


# @numba_driver.jit([(int32[:, :], int32[:, :], int32[:], int32, int32),
#                    (float32[:, :], float32[:, :], int32[:], int32, int32)])
@numba_driver.jit
def reset_when_done_2d(data, ref, done, feature_dim, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if force_reset > 0.5 or done[env_id] > 0.5:
        if tid < feature_dim:
            data[env_id, tid] = ref[env_id, tid]


# @numba_driver.jit([(int32[:, :, :], int32[:, :, :], int32[:], int32, int32, int32),
#                    (float32[:, :, :], float32[:, :, :], int32[:], int32, int32, int32)])
@numba_driver.jit
def reset_when_done_3d(data, ref, done, agent_dim, feature_dim, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if force_reset > 0.5 or done[env_id] > 0.5:
        if tid < agent_dim:
            for i in range(feature_dim):
                data[env_id, tid, i] = ref[env_id, tid, i]


@numba_driver.jit((int32[::1], int32[::1], int32))
def undo_done_flag_and_reset_timestep(done, timestep, force_reset):
    agent_id = numba_driver.threadIdx.x
    env_id = numba_driver.blockIdx.x
    if force_reset > 0.5 or done[env_id] > 0.5:
        if agent_id == 0:
            done[env_id] = 0
            timestep[env_id] = 0
