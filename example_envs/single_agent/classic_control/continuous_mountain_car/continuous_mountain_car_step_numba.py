import numba.cuda as numba_driver
import math


@numba_driver.jit
def _clip(v, min, max):
    if v < min:
        return min
    if v > max:
        return max
    return v


@numba_driver.jit
def NumbaClassicControlContinuousMountainCarEnvStep(
        state_arr,
        action_arr,
        done_arr,
        reward_arr,
        observation_arr,
        min_action,
        max_action,
        min_position,
        max_position,
        max_speed,
        goal_position,
        goal_velocity,
        power,
        env_timestep_arr,
        episode_length):

    kEnvId = numba_driver.blockIdx.x
    kThisAgentId = numba_driver.threadIdx.x

    assert kThisAgentId == 0, "We only have one agent per environment"

    env_timestep_arr[kEnvId] += 1

    assert 0 < env_timestep_arr[kEnvId] <= episode_length

    action = action_arr[kEnvId, kThisAgentId, 0]

    position = state_arr[kEnvId, kThisAgentId, 0]
    velocity = state_arr[kEnvId, kThisAgentId, 1]
    force = _clip(action, min_action, max_action)

    velocity += force * power - 0.0025 * math.cos(3 * position)
    velocity = _clip(velocity, -max_speed, max_speed)

    position += velocity
    position = _clip(position, min_position, max_position)
    if position == min_position and velocity < 0:
        velocity = 0

    state_arr[kEnvId, kThisAgentId, 0] = position
    state_arr[kEnvId, kThisAgentId, 1] = velocity

    observation_arr[kEnvId, kThisAgentId, 0] = state_arr[kEnvId, kThisAgentId, 0]
    observation_arr[kEnvId, kThisAgentId, 1] = state_arr[kEnvId, kThisAgentId, 1]

    terminated = bool(
        position >= goal_position and velocity >= goal_velocity
    )

    rew = 0.0
    if terminated:
        rew = 100.0
    rew -= math.pow(action, 2) * 0.1
    reward_arr[kEnvId, kThisAgentId] = rew

    if env_timestep_arr[kEnvId] == episode_length or terminated:
        done_arr[kEnvId] = 1