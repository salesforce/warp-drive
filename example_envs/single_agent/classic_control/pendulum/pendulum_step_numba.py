import numba
import numba.cuda as numba_driver
import numpy as np
import math

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

max_speed = 8
max_torque = 2.0
dt = 0.05
g = 9.81
m = 1.0
l = 1.0

@numba_driver.jit
def _clip(v, min, max):
    if v < min:
        return min
    if v > max:
        return max
    return v


@numba_driver.jit
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


@numba_driver.jit
def NumbaClassicControlPendulumEnvStep(
        state_arr,
        action_arr,
        done_arr,
        reward_arr,
        observation_arr,
        env_timestep_arr,
        episode_length):

    kEnvId = numba_driver.blockIdx.x
    kThisAgentId = numba_driver.threadIdx.x

    assert kThisAgentId == 0, "We only have one agent per environment"

    env_timestep_arr[kEnvId] += 1

    assert 0 < env_timestep_arr[kEnvId] <= episode_length

    action = action_arr[kEnvId, kThisAgentId, 0]

    u = _clip(action, -max_torque, max_torque)

    th = state_arr[kEnvId, kThisAgentId, 0]
    thdot = state_arr[kEnvId, kThisAgentId, 1]

    costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

    newthdot = thdot + (3 * g / (2 * l) * math.sin(th) + 3.0 / (m * l ** 2) * u) * dt
    newthdot = _clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt

    state_arr[kEnvId, kThisAgentId, 0] = newth
    state_arr[kEnvId, kThisAgentId, 1] = newthdot

    observation_arr[kEnvId, kThisAgentId, 0] = math.cos(newth)
    observation_arr[kEnvId, kThisAgentId, 1] = math.sin(newth)
    observation_arr[kEnvId, kThisAgentId, 2] = newthdot

    reward_arr[kEnvId, kThisAgentId] = -costs

    if env_timestep_arr[kEnvId] == episode_length:
        done_arr[kEnvId] = 1
