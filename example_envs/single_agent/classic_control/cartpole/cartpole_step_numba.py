import numba.cuda as numba_driver
import math


@numba_driver.jit
def NumbaClassicControlCartPoleEnvStep(
        state_arr,
        action_arr,
        done_arr,
        reward_arr,
        observation_arr,
        gravity,
        masspole,
        total_mass,
        length,
        polemass_length,
        force_mag,
        tau,
        theta_threshold_radians,
        x_threshold,
        env_timestep_arr,
        episode_length):

    kEnvId = numba_driver.blockIdx.x
    kThisAgentId = numba_driver.threadIdx.x

    assert kThisAgentId == 0, "We only have one agent per environment"

    env_timestep_arr[kEnvId] += 1

    assert 0 < env_timestep_arr[kEnvId] <= episode_length

    reward_arr[kEnvId, kThisAgentId] = 0.0

    action = action_arr[kEnvId, kThisAgentId, 0]

    x = state_arr[kEnvId, kThisAgentId, 0]
    x_dot = state_arr[kEnvId, kThisAgentId, 1]
    theta = state_arr[kEnvId, kThisAgentId, 2]
    theta_dot = state_arr[kEnvId, kThisAgentId, 3]

    if action > 0.5:
        force = force_mag
    else:
        force = -force_mag

    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + polemass_length * theta_dot ** 2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
            length * (4.0 / 3.0 - masspole * costheta ** 2 / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    # we use kinematics_integrator == "euler", same as that in the original gym code
    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc

    state_arr[kEnvId, kThisAgentId, 0] = x
    state_arr[kEnvId, kThisAgentId, 1] = x_dot
    state_arr[kEnvId, kThisAgentId, 2] = theta
    state_arr[kEnvId, kThisAgentId, 3] = theta_dot

    observation_arr[kEnvId, kThisAgentId, 0] = state_arr[kEnvId, kThisAgentId, 0]
    observation_arr[kEnvId, kThisAgentId, 1] = state_arr[kEnvId, kThisAgentId, 1]
    observation_arr[kEnvId, kThisAgentId, 2] = state_arr[kEnvId, kThisAgentId, 2]
    observation_arr[kEnvId, kThisAgentId, 3] = state_arr[kEnvId, kThisAgentId, 3]

    terminated = bool(
        x < -x_threshold
        or x > x_threshold
        or theta < -theta_threshold_radians
        or theta > theta_threshold_radians
    )

    # as long as not reset, we assign reward 1. This is consistent with original cartpole logic
    reward_arr[kEnvId, kThisAgentId] = 1.0

    if env_timestep_arr[kEnvId] == episode_length or terminated:
        done_arr[kEnvId] = 1
