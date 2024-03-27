import numpy as np
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

from example_envs.single_agent.base import SingleAgentEnv, map_to_single_agent, get_action_for_single_agent
from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS


class ClassicControlContinuousMountainCarEnv(SingleAgentEnv):

    name = "ClassicControlContinuousMountainCarEnv"

    def __init__(self, episode_length, env_backend="cpu", reset_pool_size=0, seed=None):
        super().__init__(episode_length, env_backend, reset_pool_size, seed=seed)

        self.gym_env = Continuous_MountainCarEnv()

        self.action_space = map_to_single_agent(self.gym_env.action_space)
        self.observation_space = map_to_single_agent(self.gym_env.observation_space)

    def step(self, action=None):
        self.timestep += 1
        action = get_action_for_single_agent(action)
        state, reward, terminated, _, _ = self.gym_env.step(action)

        obs = map_to_single_agent(state)
        rew = map_to_single_agent(reward)
        done = {"__all__": self.timestep >= self.episode_length or terminated}
        info = {}

        return obs, rew, done, info

    def reset(self):
        self.timestep = 0
        if self.reset_pool_size < 2:
            # we use a fixed initial state all the time
            initial_state, _ = self.gym_env.reset(seed=self.seed)
        else:
            initial_state, _ = self.gym_env.reset(seed=None)
        obs = map_to_single_agent(initial_state)

        return obs


class CUDAClassicControlContinuousMountainCarEnv(ClassicControlContinuousMountainCarEnv, CUDAEnvironmentContext):

    def get_data_dictionary(self):
        data_dict = DataFeed()
        initial_state, _ = self.gym_env.reset(seed=self.seed)

        if self.reset_pool_size < 2:
            data_dict.add_data(
                name="state",
                data=np.atleast_2d(initial_state),
                save_copy_and_apply_at_reset=True,
            )
        else:
            data_dict.add_data(
                name="state",
                data=np.atleast_2d(initial_state),
                save_copy_and_apply_at_reset=False,
            )

        data_dict.add_data_list(
            [
                ("min_action", self.gym_env.min_action),
                ("max_action", self.gym_env.max_action),
                ("min_position", self.gym_env.min_position),
                ("max_position", self.gym_env.max_position),
                ("max_speed", self.gym_env.max_speed),
                ("goal_position", self.gym_env.goal_position),
                ("goal_velocity", self.gym_env.goal_velocity),
                ("power", self.gym_env.power),
            ]
        )
        return data_dict

    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        return tensor_dict

    def get_reset_pool_dictionary(self):
        reset_pool_dict = DataFeed()
        if self.reset_pool_size >= 2:
            state_reset_pool = []
            for _ in range(self.reset_pool_size):
                initial_state, _ = self.gym_env.reset(seed=None)
                state_reset_pool.append(np.atleast_2d(initial_state))
            state_reset_pool = np.stack(state_reset_pool, axis=0)
            assert len(state_reset_pool.shape) == 3 and state_reset_pool.shape[2] == 2

            reset_pool_dict.add_pool_for_reset(name="state_reset_pool",
                                               data=state_reset_pool,
                                               reset_target="state")
        return reset_pool_dict

    def step(self, actions=None):
        self.timestep += 1
        args = [
            "state",
            _ACTIONS,
            "_done_",
            _REWARDS,
            _OBSERVATIONS,
            "min_action",
            "max_action",
            "min_position",
            "max_position",
            "max_speed",
            "goal_position",
            "goal_velocity",
            "power",
            "_timestep_",
            ("episode_length", "meta"),
        ]
        if self.env_backend == "numba":
            self.cuda_step[
                self.cuda_function_manager.grid, self.cuda_function_manager.block
            ](*self.cuda_step_function_feed(args))
        else:
            raise Exception("CUDAClassicControlContinuousMountainCarEnv expects env_backend = 'numba' ")
