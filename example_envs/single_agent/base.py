import numpy as np


class SingleAgentEnv:

    def __init__(self, episode_length=500, env_backend="cpu", reset_pool_size=0, seed=None):
        """

        :param episode_length:
        :param env_backend: "cpu" or "numba" ("pycuda" is not supported for SingleAgentEnv)
        :param reset_pool_size: if reset_pool_size < 2, we assume the reset is using a default fixed one for all envs
        """
        self.num_agents = 1

        self.agents = {}
        for agent_id in range(self.num_agents):
            self.agents[agent_id] = True

        assert episode_length > 0
        self.episode_length = episode_length

        self.action_space = None
        self.observation_space = None
        self.timestep = None

        self.env_backend = env_backend
        self.reset_pool_size = reset_pool_size

        # Seeding
        self.seed = seed


def map_to_single_agent(val):
    return {0: val}


def get_action_for_single_agent(action):
    assert isinstance(action, dict)
    assert len(action) == 1
    return action[0]