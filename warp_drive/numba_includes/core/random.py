from numba import cuda as numba_driver
from numba import float32, int32, boolean, from_dtype
from numba.cuda.random import init_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np

kEps = 1.0e-8
xoroshiro128p_type = from_dtype(np.dtype([("s0", np.uint64), ("s1", np.uint64)], align=True))


@numba_driver.jit(int32(float32[:, :, ::1], float32, int32, int32, int32), device=True, inline=True)
def search_index(distr, p, env_id, agent_id, r):
    left = 0
    right = r

    while left <= right:
        mid = left + int((right - left) / 2)
        if abs(distr[env_id, agent_id, mid] - p) < kEps:
            return mid
        elif distr[env_id, agent_id, mid] < p:
            left = mid + 1
        else:
            right = mid - 1
    if left > r:
        return r
    else:
        return left


def init_random(rng_states, seed):
    init_xoroshiro128p_states(states=rng_states, seed=seed)


@numba_driver.jit((xoroshiro128p_type[::1], float32[:, :, ::1], int32[:, :, ::1], float32[:, :, ::1], int32, int32))
def sample_actions(rng_states, distr, action_indices, cum_distr, num_actions, use_argmax):
    env_id = numba_driver.blockIdx.x
    # Block id in a 1D grid
    agent_id = numba_driver.threadIdx.x
    posidx = numba_driver.grid(1)
    if posidx >= rng_states.shape[0]:
        return

    if use_argmax > 0.5:
        max_dist = distr[env_id, agent_id, 0]
        max_ind = 0
        for i in range(1, num_actions):
            if max_dist < distr[env_id, agent_id, i]:
                max_dist = distr[env_id, agent_id, i]
                max_ind = i
        action_indices[env_id, agent_id, 0] = max_ind
        return

    p = xoroshiro128p_uniform_float32(rng_states, posidx)

    cum_distr[env_id, agent_id, 0] = distr[env_id, agent_id, 0]

    for i in range(1, num_actions):
        cum_distr[env_id, agent_id, i] = (
            distr[env_id, agent_id, i] + cum_distr[env_id, agent_id, i - 1]
        )

    ind = search_index(cum_distr, p, env_id, agent_id, num_actions - 1)
    # action_indices in the shape of [n_env, n_agent, 1]
    action_indices[env_id, agent_id, 0] = ind
