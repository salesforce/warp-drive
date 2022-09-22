import numba.cuda as numba_driver


@numba_driver.jit
def cuda_increment(data, num_agents):
    env_id = numba_driver.blockIdx.x
    agent_id = numba_driver.threadIdx.x
    if agent_id < num_agents:
        increment = env_id + agent_id
        data[env_id, agent_id] += increment
