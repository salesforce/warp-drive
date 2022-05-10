from numba import int32, float32
from numba import cuda as numba_driver
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from numba_includes.env_config import *


kEps = 1.0e-8


class NumbaRandomService:

    def __init__(self, seed=None):
        self.rng_states = create_xoroshiro128p_states(wkNumberEnvs * wkNumberAgents, seed=seed)

    @numba_driver.jit(int32(float32[::1], float32, int32, int32), device=True)
    def search_index(self, distr, p, l, r):
        left = l
        right = r

        while left <= right:
            mid = left + (right - left) / 2
            if abs(distr[mid] - p) < kEps:
                return mid - l
            elif distr[mid] < p:
                left = mid + 1
            else:
                right = mid - 1
        if left > r:
            return r - l
        else:
            return left - l

    @numba_driver.jit((float32[::1], int32[::1], float32[::1], int32, int32))
    def sample_actions(self, distr, action_indices, cum_distr, num_agents, num_actions):
        posidx = numba_driver.grid(1)
        if posidx >= wkNumberEnvs * num_agents:
            return

        p = xoroshiro128p_uniform_float32(self.rng_states, posidx)

        dist_index = posidx * num_actions
        cum_distr[dist_index] = distr[dist_index]

        for i in range(1, num_actions):
            cum_distr[dist_index + i] = distr[dist_index + i] + cum_distr[dist_index + i - 1]

        ind = search_index(cum_distr, p, dist_index, dist_index + num_actions - 1)
        action_indices[posidx] = ind

