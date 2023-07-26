from numba import cuda as numba_driver
from numba import float32, int32, from_dtype
from numba.cuda.random import init_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np

xoroshiro128p_type = from_dtype(np.dtype([("s0", np.uint64), ("s1", np.uint64)], align=True))


def init_random_for_reset(rng_states, seed):
    init_xoroshiro128p_states(states=rng_states, seed=seed)


# @numba_driver.jit([(int32[:], int32[:], int32[:], int32),
#                    (float32[:], float32[:], int32[:], int32)])
@numba_driver.jit
def reset_when_done_1d_from_pool(rng_states, data, ref, done, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if tid == 0:
        if force_reset > 0.5 or done[env_id] > 0.5:
            p = xoroshiro128p_uniform_float32(rng_states, env_id)
            ref_id_float = p * ref.shape[0]
            ref_id = int(ref_id_float)
            data[env_id] = ref[ref_id]


# @numba_driver.jit([(int32[:, :], int32[:, :], int32[:], int32, int32),
#                    (float32[:, :], float32[:, :], int32[:], int32, int32)])
@numba_driver.jit
def reset_when_done_2d_from_pool(rng_states, data, ref, done, feature_dim, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if force_reset > 0.5 or done[env_id] > 0.5:
        p = xoroshiro128p_uniform_float32(rng_states, env_id)
        ref_id_float = p * ref.shape[0]
        ref_id = int(ref_id_float)
        if tid < feature_dim:
            data[env_id, tid] = ref[ref_id, tid]


# @numba_driver.jit([(int32[:, :, :], int32[:, :, :], int32[:], int32, int32, int32),
#                    (float32[:, :, :], float32[:, :, :], int32[:], int32, int32, int32)])
@numba_driver.jit
def reset_when_done_3d_from_pool(rng_states, data, ref, done, agent_dim, feature_dim, force_reset):
    env_id = numba_driver.blockIdx.x
    tid = numba_driver.threadIdx.x
    if force_reset > 0.5 or done[env_id] > 0.5:
        p = xoroshiro128p_uniform_float32(rng_states, env_id)
        ref_id_float = p * ref.shape[0]
        ref_id = int(ref_id_float)
        if tid < agent_dim:
            for i in range(feature_dim):
                data[env_id, tid, i] = ref[ref_id, tid, i]