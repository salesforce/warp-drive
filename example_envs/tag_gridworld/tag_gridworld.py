# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from gym import spaces
from gym.utils import seeding

# seeding code from https://github.com/openai/gym/blob/master/gym/utils/seeding.py
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_LOC_X = "loc_x"
_LOC_Y = "loc_y"


class TagGridWorld:
    """
    The game of tag on a 2D square grid plane.
    This is a simplified version of the continuous tag.
    There are a number of taggers trying to tag 1 runner.
    The taggers want to catch the runner. Once the runner is tagged, the game is over.
    """

    def __init__(
        self,
        num_taggers=10,
        grid_length=10,
        episode_length=100,
        starting_location_x=None,
        starting_location_y=None,
        seed=None,
        wall_hit_penalty=0.1,
        tag_reward_for_tagger=10.0,
        tag_penalty_for_runner=2.0,
        step_cost_for_tagger=0.01,
        use_full_observation=True,
    ):
        """
        :param num_taggers (int): the total number of taggers. In this env,
            num_runner = 1
        :param grid_length (int): the world is a square with grid_length,
        :param episode_length (int): episode length
        :param starting_location_x ([ndarray], optional): starting x locations
            of the agents. If None, start from center
        :param starting_location_y ([ndarray], optional): starting y locations
            of the agents. If None, start from center
        :param seed: seeding parameter
        :param wall_hit_penalty (float): penalty of hitting the wall
        :param tag_reward_for_tagger (float): tag reward for taggers
        :param tag_penalty_for_runner (float): tag penalty for runner
        :param step_cost_for_tagger (float): penalty for each step
        :param use_full_observation (bool): boolean indicating whether to
            include all the agents' data in the use_full_observation or
            just the nearest neighbor. Defaults to True.
        """
        assert num_taggers > 0
        self.num_taggers = num_taggers
        # there is also (only) one runner
        self.num_agents = self.num_taggers + 1

        assert episode_length > 0
        self.episode_length = episode_length

        self.grid_length = grid_length

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        self.agent_type = {}
        self.taggers = {}
        self.runners = {}
        for agent_id in range(self.num_agents):
            if agent_id < self.num_taggers:
                self.agent_type[agent_id] = 0  # Tagger
                self.taggers[agent_id] = True
            else:
                self.agent_type[agent_id] = 1  # Runner
                self.runners[agent_id] = True

        if starting_location_x is None:
            assert starting_location_y is None
            # taggers are starting in the center of the grid
            # and the runner in the corner [0, 0]
            starting_location_x = int(0.5 * self.grid_length) * np.ones(self.num_agents)
            starting_location_x[-1] = 0
            starting_location_y = int(0.5 * self.grid_length) * np.ones(self.num_agents)
            starting_location_y[-1] = 0
        else:
            assert len(starting_location_x) == self.num_agents
            assert len(starting_location_y) == self.num_agents

        self.starting_location_x = starting_location_x
        self.starting_location_y = starting_location_y

        self.step_actions = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper

        self.action_space = {
            agent_id: spaces.Discrete(len(self.step_actions))
            for agent_id in range(self.num_agents)
        }

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # For reward computation
        self.wall_hit_penalty = wall_hit_penalty
        self.tag_reward_for_tagger = tag_reward_for_tagger
        self.tag_penalty_for_runner = tag_penalty_for_runner
        self.step_cost_for_tagger = step_cost_for_tagger
        self.reward_penalty = np.zeros(self.num_agents)
        self.reward_tag = np.zeros(self.num_agents)
        self.use_full_observation = use_full_observation

    name = "TagGridWorld"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_global_state(self, key=None, value=None, t=None, dtype=None):
        assert key is not None
        if dtype is None:
            dtype = np.float32

        # If no values are passed, set everything to zeros.
        if key not in self.global_state:
            self.global_state[key] = np.zeros(
                (self.episode_length + 1, self.num_agents), dtype=dtype
            )

        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == self.global_state[key].shape[1]

            self.global_state[key][t] = value

    def update_state(self, actions_x, actions_y):
        loc_x_prev_t = self.global_state[_LOC_X][self.timestep - 1]
        loc_y_prev_t = self.global_state[_LOC_Y][self.timestep - 1]

        loc_x_curr_t = loc_x_prev_t + actions_x
        loc_y_curr_t = loc_y_prev_t + actions_y

        clipped_loc_x_curr_t = np.clip(loc_x_curr_t, 0, self.grid_length)
        clipped_loc_y_curr_t = np.clip(loc_y_curr_t, 0, self.grid_length)

        # Penalize reward if agents hit the walls
        self.reward_penalty = (
            -1.0
            * self.wall_hit_penalty
            * (
                (loc_x_curr_t != clipped_loc_x_curr_t)
                | (loc_y_curr_t != clipped_loc_y_curr_t)
            )
        )

        self.set_global_state(key=_LOC_X, value=clipped_loc_x_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Y, value=clipped_loc_y_curr_t, t=self.timestep)

        tag = (
            (clipped_loc_x_curr_t[: self.num_taggers] == clipped_loc_x_curr_t[-1])
            & (clipped_loc_y_curr_t[: self.num_taggers] == clipped_loc_y_curr_t[-1])
        ).any()
        # if tagged
        if tag:
            self.reward_tag[: self.num_taggers] = self.tag_reward_for_tagger
            self.reward_tag[-1] = -1.0 * self.tag_penalty_for_runner
        else:
            self.reward_tag[: self.num_taggers] = -1.0 * self.step_cost_for_tagger
            self.reward_tag[-1] = 1.0 * self.step_cost_for_tagger

        reward = self.reward_tag + self.reward_penalty
        rew = {}
        for agent_id, r in enumerate(reward):
            rew[agent_id] = r

        return rew, tag

    def generate_observation(self):
        obs = {}
        if self.use_full_observation:
            common_obs = None
            for feature in [
                _LOC_X,
                _LOC_Y,
            ]:
                if common_obs is None:
                    common_obs = self.global_state[feature][self.timestep]
                else:
                    common_obs = np.vstack(
                        (common_obs, self.global_state[feature][self.timestep])
                    )
            normalized_common_obs = common_obs / self.grid_length

            agent_types = np.array(
                [self.agent_type[agent_id] for agent_id in range(self.num_agents)]
            )

            for agent_id in range(self.num_agents):
                agent_indicators = np.zeros(self.num_agents)
                agent_indicators[agent_id] = 1
                obs[agent_id] = np.concatenate(
                    [
                        np.vstack(
                            (normalized_common_obs, agent_types, agent_indicators)
                        ).reshape(-1),
                        np.array([float(self.timestep) / self.episode_length]),
                    ]
                )
        else:
            for agent_id in range(self.num_agents):
                feature_list = []
                for feature in [
                    _LOC_X,
                    _LOC_Y,
                ]:
                    feature_list.append(
                        self.global_state[feature][self.timestep][agent_id]
                        / self.grid_length
                    )
                if agent_id < self.num_agents - 1:
                    for feature in [
                        _LOC_X,
                        _LOC_Y,
                    ]:
                        feature_list.append(
                            self.global_state[feature][self.timestep][-1]
                            / self.grid_length
                        )
                else:
                    dist_array = None
                    for feature in [
                        _LOC_X,
                        _LOC_Y,
                    ]:
                        if dist_array is None:
                            dist_array = np.square(
                                self.global_state[feature][self.timestep][:-1]
                                - self.global_state[feature][self.timestep][-1]
                            )
                        else:
                            dist_array += np.square(
                                self.global_state[feature][self.timestep][:-1]
                                - self.global_state[feature][self.timestep][-1]
                            )
                    min_agent_id = np.argmin(dist_array)
                    for feature in [
                        _LOC_X,
                        _LOC_Y,
                    ]:
                        feature_list.append(
                            self.global_state[feature][self.timestep][min_agent_id]
                            / self.grid_length
                        )
                feature_list += [
                    self.agent_type[agent_id],
                    float(self.timestep) / self.episode_length,
                ]
                obs[agent_id] = np.array(feature_list)
        return obs

    def reset(self):
        # Reset time to the beginning
        self.timestep = 0

        # Re-initialize the global state
        self.global_state = {}
        self.set_global_state(
            key=_LOC_X, value=self.starting_location_x, t=self.timestep, dtype=np.int32
        )
        self.set_global_state(
            key=_LOC_Y, value=self.starting_location_y, t=self.timestep, dtype=np.int32
        )
        return self.generate_observation()

    def step(
        self,
        actions=None,
    ):
        self.timestep += 1
        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        actions_x = np.array(
            [
                self.step_actions[actions[agent_id]][0]
                for agent_id in range(self.num_agents)
            ]
        )
        actions_y = np.array(
            [
                self.step_actions[actions[agent_id]][1]
                for agent_id in range(self.num_agents)
            ]
        )

        rew, tag = self.update_state(actions_x, actions_y)
        obs = self.generate_observation()
        done = {"__all__": self.timestep >= self.episode_length or tag}
        info = {}

        return obs, rew, done, info


class CUDATagGridWorld(TagGridWorld, CUDAEnvironmentContext):
    """
    CUDA version of the TagGridWorld environment.
    Note: this class subclasses the Python environment class TagGridWorld,
    and also the  CUDAEnvironmentContext
    """

    def get_data_dictionary(self):
        data_dict = DataFeed()
        for feature in [
            _LOC_X,
            _LOC_Y,
        ]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
                log_data_across_episode=True,
            )
        data_dict.add_data_list(
            [
                ("wall_hit_penalty", self.wall_hit_penalty),
                ("tag_reward_for_tagger", self.tag_reward_for_tagger),
                ("tag_penalty_for_runner", self.tag_penalty_for_runner),
                ("step_cost_for_tagger", self.step_cost_for_tagger),
                ("use_full_observation", self.use_full_observation),
                ("world_boundary", self.grid_length),
            ]
        )
        return data_dict

    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        return tensor_dict

    def step(self, actions=None):
        self.timestep += 1
        args = [
            _LOC_X,
            _LOC_Y,
            _ACTIONS,
            "_done_",
            _REWARDS,
            _OBSERVATIONS,
            "wall_hit_penalty",
            "tag_reward_for_tagger",
            "tag_penalty_for_runner",
            "step_cost_for_tagger",
            "use_full_observation",
            "world_boundary",
            "_timestep_",
            ("episode_length", "meta"),
        ]
        self.cuda_step(
            *self.cuda_step_function_feed(args),
            block=self.cuda_function_manager.block,
            grid=self.cuda_function_manager.grid,
        )
