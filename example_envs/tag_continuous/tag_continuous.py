# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import copy
import heapq

import numpy as np
from gym import spaces
from gym.utils import seeding

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_LOC_X = "loc_x"
_LOC_Y = "loc_y"
_SP = "speed"
_DIR = "direction"
_ACC = "acceleration"
_SIG = "still_in_the_game"


class TagContinuous(CUDAEnvironmentContext):
    """
    The game of tag on a continuous circular 2D space.
    There are some taggers trying to tag several runners.
    The taggers want to get as close as possible to the runner, while the runner
    wants to get as far away from them as possible.
    Once a runner is tagged, he exits the game if runner_exits_game_after_tagged is True
    otherwise he continues to run around (and the tagger can catch him again)
    """

    def __init__(
        self,
        num_taggers=1,
        num_runners=10,
        grid_length=10.0,
        episode_length=100,
        starting_location_x=None,
        starting_location_y=None,
        starting_directions=None,
        seed=None,
        max_speed=1.0,
        skill_level_runner=1.0,
        skill_level_tagger=1.0,
        max_acceleration=1.0,
        min_acceleration=-1.0,
        max_turn=np.pi / 2,
        min_turn=-np.pi / 2,
        num_acceleration_levels=10,
        num_turn_levels=10,
        edge_hit_penalty=-0.0,
        use_full_observation=True,
        num_other_agents_observed=2,
        tagging_distance=0.01,
        tag_reward_for_tagger=1.0,
        step_penalty_for_tagger=-0.0,
        tag_penalty_for_runner=-1.0,
        step_reward_for_runner=0.0,
        end_of_game_reward_for_runner=1.0,
        runner_exits_game_after_tagged=True,
        use_cuda=False,
    ):
        """
        Args:
            num_taggers (int, optional): [number of taggers in the environment].
                Defaults to 1.
            num_runners (int, optional): [number of taggers in the environment].
                Defaults to 10.
            grid_length (float, optional): [length of the square grid]. Defaults to 10.0
            episode_length (int, optional): [episode length]. Defaults to 100.
            starting_location_x ([ndarray], optional): [starting x locations of the
                agents]. Defaults to None.
            starting_location_y ([ndarray], optional): [starting y locations of the
            agents]. Defaults to None.
            starting_directions ([ndarray], optional): starting orientations
                in [0, 2*pi]. Defaults to None.
            seed ([type], optional): [seeding parameter]. Defaults to None.
            max_speed (float, optional): [max speed of the agents]. Defaults to 1.0
            skill_level_runner (float, optional): [runner skill level;
                this essentially is a multiplier to the max_speed].
                Defaults to 1.0
            skill_level_tagger (float, optional): [tagger skill level]. Defaults to 1.0
            max_acceleration (float, optional): [the max acceleration]. Defaults to 1.0.
            min_acceleration (float, optional): [description]. Defaults to -1.0
            max_turn ([type], optional): [description]. Defaults to np.pi/2.
            min_turn ([type], optional): [description]. Defaults to -np.pi/2.
            num_acceleration_levels (int, optional): [number of acceleration actions
                uniformly spaced between max and min acceleration]. Defaults to 10.
            num_turn_levels (int, optional): [number of turn actions uniformly spaced
                between max and min turns]. Defaults to 10.
            edge_hit_penalty (float, optional): [penalty for hitting the edge (wall)].
                Defaults to -0.0.
            use_full_observation (bool, optional): [boolean indicating whether to
                include all the agents' data in the observation or just the nearest
                neighbors]. Defaults to True.
            num_other_agents_observed (int, optional): [number of nearest neighbors
                in the obs (only takes effect when use_full_observation is False)].
                Defaults to 2.
            tagging_distance (float, optional): [margin between a
                tagger and runner to consider the runner as 'tagged'. This multiplies
                on top of the grid length]. Defaults to 0.01.
            tag_reward_for_tagger (float, optional): [positive reward for the tagger
                upon tagging a runner]. Defaults to 1.0
            step_penalty_for_tagger (float, optional): [penalty for every step
                the game goes on]. Defaults to -0.0.
            tag_penalty_for_runner (float, optional): [negative reward for getting
                tagged]. Defaults to -1.0
            step_reward_for_runner (float, optional): [reward for every step the
                runner isn't tagged]. Defaults to 0.0.
            end_of_game_reward_for_runner (float, optional): [reward at the end of
                the game for a runner that isn't tagged]. Defaults to 1.0.
            runner_exits_game_after_tagged (bool, optional): [boolean indicating
                whether runners exit the game after getting tagged or can remain in and
                continue to get tagged]. Defaults to True.
            use_cuda (bool, optional): [boolean to indicate whether to use the CPU
                or the GPU. (cuda) for stepping through the environment].
                Defaults to False.
        """
        super().__init__()

        self.float_dtype = np.float32
        self.int_dtype = np.int32
        # small number to prevent indeterminate cases
        self.eps = self.float_dtype(1e-10)

        assert num_taggers > 0
        self.num_taggers = num_taggers

        assert num_runners > 0
        self.num_runners = num_runners

        self.num_agents = self.num_taggers + self.num_runners

        assert episode_length > 0
        self.episode_length = episode_length

        # Square 2D grid
        assert grid_length > 0
        self.grid_length = self.float_dtype(grid_length)
        self.grid_diagonal = self.grid_length * np.sqrt(2)

        # Penalty for hitting the edges
        assert edge_hit_penalty <= 0
        self.edge_hit_penalty = self.float_dtype(edge_hit_penalty)

        # Seeding
        self.np_random = np.random
        if seed is not None:
            self.seed(seed)

        # Starting taggers
        taggers = self.np_random.choice(
            np.arange(self.num_agents), self.num_taggers, replace=False
        )

        self.agent_type = {}
        self.taggers = {}
        self.runners = {}
        for agent_id in range(self.num_agents):
            if agent_id in set(taggers):
                self.agent_type[agent_id] = 1  # Tagger
                self.taggers[agent_id] = True
            else:
                self.agent_type[agent_id] = 0  # Runner
                self.runners[agent_id] = True

        if starting_location_x is None:
            assert starting_location_y is None

            starting_location_x = self.grid_length * self.np_random.rand(
                self.num_agents
            )
            starting_location_y = self.grid_length * self.np_random.rand(
                self.num_agents
            )
        else:
            assert len(starting_location_x) == self.num_agents
            assert len(starting_location_y) == self.num_agents

        self.starting_location_x = starting_location_x
        self.starting_location_y = starting_location_y

        if starting_directions is None:
            starting_directions = self.np_random.choice(
                [0, np.pi / 2, np.pi, np.pi * 3 / 2], self.num_agents, replace=True
            )
        else:
            assert len(starting_directions) == self.num_agents
        self.starting_directions = starting_directions

        # Set the max speed level
        self.max_speed = self.float_dtype(max_speed)

        # All agents start with 0 speed and acceleration
        self.starting_speeds = np.zeros(self.num_agents, dtype=self.float_dtype)
        self.starting_accelerations = np.zeros(self.num_agents, dtype=self.float_dtype)

        assert num_acceleration_levels >= 0
        assert num_turn_levels >= 0

        # The num_acceleration and num_turn levels refer to the number of
        # uniformly-spaced levels between (min_acceleration and max_acceleration)
        # and (min_turn and max_turn), respectively.
        self.num_acceleration_levels = num_acceleration_levels
        self.num_turn_levels = num_turn_levels
        self.max_acceleration = self.float_dtype(max_acceleration)
        self.min_acceleration = self.float_dtype(min_acceleration)

        self.max_turn = self.float_dtype(max_turn)
        self.min_turn = self.float_dtype(min_turn)

        # Acceleration actions
        self.acceleration_actions = np.linspace(
            self.min_acceleration, self.max_acceleration, self.num_acceleration_levels
        )
        # Add action 0 - this will be the no-op, or 0 acceleration
        self.acceleration_actions = np.insert(self.acceleration_actions, 0, 0).astype(
            self.float_dtype
        )

        # Turn actions
        self.turn_actions = np.linspace(
            self.min_turn, self.max_turn, self.num_turn_levels
        )
        # Add action 0 - this will be the no-op, or 0 turn
        self.turn_actions = np.insert(self.turn_actions, 0, 0).astype(self.float_dtype)

        # Tagger and runner agent skill levels.
        # Skill levels multiply on top of the acceleration levels
        self.skill_levels = [
            self.agent_type[agent_id] * self.float_dtype(skill_level_tagger)
            + (1 - self.agent_type[agent_id]) * self.float_dtype(skill_level_runner)
            for agent_id in range(self.num_agents)
        ]

        # Does the runner exit the game or continue to play after getting tagged?
        self.runner_exits_game_after_tagged = runner_exits_game_after_tagged

        # These will be set during reset (see below)
        self.timestep = None
        self.global_state = None

        # Defining observation and action spaces
        self.observation_space = None  # Note: this will be set via the env_wrapper
        self.action_space = {
            agent_id: spaces.MultiDiscrete(
                (len(self.acceleration_actions), len(self.turn_actions))
            )
            for agent_id in range(self.num_agents)
        }
        # Used in generate_observation()
        # When use_full_observation is True, then all the agents will have info of
        # all the other agents, otherwise, each agent will only have info of
        # its k-nearest agents (k = num_other_agents_observed)
        self.use_full_observation = use_full_observation
        self.init_obs = None  # Will be set later in generate_observation()

        assert num_other_agents_observed <= self.num_agents
        self.num_other_agents_observed = num_other_agents_observed

        # Distance margin between agents for non-zero rewards
        # If a tagger is closer than this to a runner, the tagger
        # gets a positive reward, and the runner a negative reward
        assert 0 <= tagging_distance <= 1
        self.distance_margin_for_reward = (tagging_distance * self.grid_length).astype(
            self.float_dtype
        )

        # Rewards and penalties
        assert tag_reward_for_tagger >= 0
        self.tag_reward_for_tagger = self.float_dtype(tag_reward_for_tagger)
        assert step_penalty_for_tagger <= 0
        self.step_penalty_for_tagger = self.float_dtype(step_penalty_for_tagger)

        assert tag_penalty_for_runner <= 0
        self.tag_penalty_for_runner = self.float_dtype(tag_penalty_for_runner)
        assert step_reward_for_runner >= 0
        self.step_reward_for_runner = self.float_dtype(step_reward_for_runner)

        self.step_rewards = [
            self.agent_type[agent_id] * self.step_penalty_for_tagger
            + (1 - self.agent_type[agent_id]) * self.step_reward_for_runner
            for agent_id in range(self.num_agents)
        ]

        assert end_of_game_reward_for_runner >= 0
        self.end_of_game_reward_for_runner = self.float_dtype(
            end_of_game_reward_for_runner
        )

        # Note: These will be set later
        self.edge_hit_reward_penalty = None
        self.still_in_the_game = None

        # These will also be set via the env_wrapper
        # use_cuda will be set to True (by the env_wrapper), if needed
        # to be simulated on the GPU
        self.use_cuda = use_cuda

        # Copy runners dict for applying at reset
        self.runners_at_reset = copy.deepcopy(self.runners)

    name = "TagContinuous"

    def seed(self, seed=None):
        """
        Seeding the environment with a desired seed
        Note: this uses the code in
        https://github.com/openai/gym/blob/master/gym/utils/seeding.py
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_global_state(self, key=None, value=None, t=None, dtype=None):
        """
        Set the global state for a specified key, value and timestep.
        Note: for a new key, initialize global state to all zeros.
        """
        assert key is not None
        if dtype is None:
            dtype = self.float_dtype

        # If no values are passed, set everything to zeros.
        if key not in self.global_state:
            self.global_state[key] = np.zeros(
                (self.episode_length + 1, self.num_agents), dtype=dtype
            )

        if t is not None and value is not None:
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == self.global_state[key].shape[1]

            self.global_state[key][t] = value

    def update_state(self, delta_accelerations, delta_turns):
        """
        Note: 'update_state' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (update_state)
        is part of the step() function!

        The logic below mirrors (part of) the step function in CUDA.
        """
        loc_x_prev_t = self.global_state[_LOC_X][self.timestep - 1]
        loc_y_prev_t = self.global_state[_LOC_Y][self.timestep - 1]
        speed_prev_t = self.global_state[_SP][self.timestep - 1]
        dir_prev_t = self.global_state[_DIR][self.timestep - 1]
        acc_prev_t = self.global_state[_ACC][self.timestep - 1]

        # Update direction and acceleration
        # Do not update location if agent is out of the game !
        dir_curr_t = (
            (dir_prev_t + delta_turns) % (2 * np.pi) * self.still_in_the_game
        ).astype(self.float_dtype)

        acc_curr_t = acc_prev_t + delta_accelerations

        # 0 <= speed <= max_speed (multiplied by the skill levels).
        # Reset acceleration to 0 when speed is outside this range
        max_speed = self.max_speed * np.array(self.skill_levels)
        speed_curr_t = self.float_dtype(
            np.clip(speed_prev_t + acc_curr_t, 0.0, max_speed) * self.still_in_the_game
        )
        acc_curr_t = acc_curr_t * (speed_curr_t > 0) * (speed_curr_t < max_speed)

        loc_x_curr_t = self.float_dtype(
            loc_x_prev_t + speed_curr_t * np.cos(dir_curr_t)
        )
        loc_y_curr_t = self.float_dtype(
            loc_y_prev_t + speed_curr_t * np.sin(dir_curr_t)
        )

        # Crossing the edge
        has_crossed_edge = ~(
            (loc_x_curr_t >= 0)
            & (loc_x_curr_t <= self.grid_length)
            & (loc_y_curr_t >= 0)
            & (loc_y_curr_t <= self.grid_length)
        )

        # Clip x and y if agent has crossed edge
        clipped_loc_x_curr_t = self.float_dtype(
            np.clip(loc_x_curr_t, 0.0, self.grid_length)
        )

        clipped_loc_y_curr_t = self.float_dtype(
            np.clip(loc_y_curr_t, 0.0, self.grid_length)
        )

        # Penalize reward if agents hit the walls
        self.edge_hit_reward_penalty = self.edge_hit_penalty * has_crossed_edge

        # Set global states
        self.set_global_state(key=_LOC_X, value=clipped_loc_x_curr_t, t=self.timestep)
        self.set_global_state(key=_LOC_Y, value=clipped_loc_y_curr_t, t=self.timestep)
        self.set_global_state(key=_SP, value=speed_curr_t, t=self.timestep)
        self.set_global_state(key=_DIR, value=dir_curr_t, t=self.timestep)
        self.set_global_state(key=_ACC, value=acc_curr_t, t=self.timestep)

    def compute_distance(self, agent1, agent2):
        """
        Note: 'compute_distance' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (compute_distance)
        is also part of the step() function!
        """
        return np.sqrt(
            (
                self.global_state[_LOC_X][self.timestep, agent1]
                - self.global_state[_LOC_X][self.timestep, agent2]
            )
            ** 2
            + (
                self.global_state[_LOC_Y][self.timestep, agent1]
                - self.global_state[_LOC_Y][self.timestep, agent2]
            )
            ** 2
        ).astype(self.float_dtype)

    def k_nearest_neighbors(self, agent_id, k):
        """
        Note: 'k_nearest_neighbors' is only used when running on CPU step() only.
        When using the CUDA step function, this Python method (k_nearest_neighbors)
        is also part of the step() function!
        """
        agent_ids_and_distances = []

        for ag_id in range(self.num_agents):
            if (ag_id != agent_id) and (self.still_in_the_game[ag_id]):
                agent_ids_and_distances += [
                    (ag_id, self.compute_distance(agent_id, ag_id))
                ]
        k_nearest_neighbor_ids_and_distances = heapq.nsmallest(
            k, agent_ids_and_distances, key=lambda x: x[1]
        )

        return [
            item[0]
            for item in k_nearest_neighbor_ids_and_distances[
                : self.num_other_agents_observed
            ]
        ]

    def generate_observation(self):
        """
        Generate and return the observations for every agent.
        """
        obs = {}

        normalized_global_obs = None
        for feature in [
            (_LOC_X, self.grid_diagonal),
            (_LOC_Y, self.grid_diagonal),
            (_SP, self.max_speed + self.eps),
            (_ACC, self.max_speed + self.eps),
            (_DIR, 2 * np.pi),
        ]:
            if normalized_global_obs is None:
                normalized_global_obs = (
                    self.global_state[feature[0]][self.timestep] / feature[1]
                )
            else:
                normalized_global_obs = np.vstack(
                    (
                        normalized_global_obs,
                        self.global_state[feature[0]][self.timestep] / feature[1],
                    )
                )
        agent_types = np.array(
            [self.agent_type[agent_id] for agent_id in range(self.num_agents)]
        )
        time = np.array([float(self.timestep) / self.episode_length])

        if self.use_full_observation:
            for agent_id in range(self.num_agents):
                # Initialize obs
                obs[agent_id] = np.concatenate(
                    [
                        np.vstack(
                            (
                                np.zeros_like(normalized_global_obs),
                                agent_types,
                                self.still_in_the_game,
                            )
                        )[
                            :,
                            [idx for idx in range(self.num_agents) if idx != agent_id],
                        ].reshape(
                            -1
                        ),  # filter out the obs for the current agent
                        np.array([0.0]),
                    ]
                )
                # Set obs for agents still in the game
                if self.still_in_the_game[agent_id]:
                    obs[agent_id] = np.concatenate(
                        [
                            np.vstack(
                                (
                                    normalized_global_obs
                                    - normalized_global_obs[:, agent_id].reshape(-1, 1),
                                    agent_types,
                                    self.still_in_the_game,
                                )
                            )[
                                :,
                                [
                                    idx
                                    for idx in range(self.num_agents)
                                    if idx != agent_id
                                ],
                            ].reshape(
                                -1
                            ),  # filter out the obs for the current agent
                            time,
                        ]
                    )
        else:  # use partial observation
            for agent_id in range(self.num_agents):
                if self.timestep == 0:
                    # Set obs to all zeros
                    obs_global_states = np.zeros(
                        (
                            normalized_global_obs.shape[0],
                            self.num_other_agents_observed,
                        )
                    )
                    obs_agent_types = np.zeros(self.num_other_agents_observed)
                    obs_still_in_the_game = np.zeros(self.num_other_agents_observed)

                    # Form the observation
                    self.init_obs = np.concatenate(
                        [
                            np.vstack(
                                (
                                    obs_global_states,
                                    obs_agent_types,
                                    obs_still_in_the_game,
                                )
                            ).reshape(-1),
                            np.array([0.0]),  # time
                        ]
                    )

                # Initialize obs to all zeros
                obs[agent_id] = self.init_obs

                # Set obs for agents still in the game
                if self.still_in_the_game[agent_id]:
                    nearest_neighbor_ids = self.k_nearest_neighbors(
                        agent_id, k=self.num_other_agents_observed
                    )
                    # For the case when the number of remaining agent ids is fewer
                    # than self.num_other_agents_observed (because agents have exited
                    # the game), we also need to pad obs wih zeros
                    obs_global_states = np.hstack(
                        (
                            normalized_global_obs[:, nearest_neighbor_ids]
                            - normalized_global_obs[:, agent_id].reshape(-1, 1),
                            np.zeros(
                                (
                                    normalized_global_obs.shape[0],
                                    self.num_other_agents_observed
                                    - len(nearest_neighbor_ids),
                                )
                            ),
                        )
                    )
                    obs_agent_types = np.hstack(
                        (
                            agent_types[nearest_neighbor_ids],
                            np.zeros(
                                (
                                    self.num_other_agents_observed
                                    - len(nearest_neighbor_ids)
                                )
                            ),
                        )
                    )
                    obs_still_in_the_game = (
                        np.hstack(
                            (
                                self.still_in_the_game[nearest_neighbor_ids],
                                np.zeros(
                                    (
                                        self.num_other_agents_observed
                                        - len(nearest_neighbor_ids)
                                    )
                                ),
                            )
                        ),
                    )

                    # Form the observation
                    obs[agent_id] = np.concatenate(
                        [
                            np.vstack(
                                (
                                    obs_global_states,
                                    obs_agent_types,
                                    obs_still_in_the_game,
                                )
                            ).reshape(-1),
                            time,
                        ]
                    )

        return obs

    def compute_reward(self):
        """
        Compute and return the rewards for each agent.
        """
        # Initialize rewards
        rew = {agent_id: 0.0 for agent_id in range(self.num_agents)}

        taggers_list = sorted(self.taggers)

        # At least one runner present
        if self.num_runners > 0:
            runners_list = sorted(self.runners)
            runner_locations_x = self.global_state[_LOC_X][self.timestep][runners_list]
            tagger_locations_x = self.global_state[_LOC_X][self.timestep][taggers_list]

            runner_locations_y = self.global_state[_LOC_Y][self.timestep][runners_list]
            tagger_locations_y = self.global_state[_LOC_Y][self.timestep][taggers_list]

            runners_to_taggers_distances = np.sqrt(
                (
                    np.repeat(runner_locations_x, self.num_taggers)
                    - np.tile(tagger_locations_x, self.num_runners)
                )
                ** 2
                + (
                    np.repeat(runner_locations_y, self.num_taggers)
                    - np.tile(tagger_locations_y, self.num_runners)
                )
                ** 2
            ).reshape(self.num_runners, self.num_taggers)

            min_runners_to_taggers_distances = np.min(
                runners_to_taggers_distances, axis=1
            )
            argmin_runners_to_taggers_distances = np.argmin(
                runners_to_taggers_distances, axis=1
            )
            nearest_tagger_ids = [
                taggers_list[idx] for idx in argmin_runners_to_taggers_distances
            ]

        # Rewards
        # Add edge hit reward penalty and the step rewards/ penalties
        for agent_id in range(self.num_agents):
            if self.still_in_the_game[agent_id]:
                rew[agent_id] += self.edge_hit_reward_penalty[agent_id]
                rew[agent_id] += self.step_rewards[agent_id]

        for idx, runner_id in enumerate(runners_list):
            if min_runners_to_taggers_distances[idx] < self.distance_margin_for_reward:

                # the runner is tagged!
                rew[runner_id] += self.tag_penalty_for_runner
                rew[nearest_tagger_ids[idx]] += self.tag_reward_for_tagger

                if self.runner_exits_game_after_tagged:
                    # Remove runner from game
                    self.still_in_the_game[runner_id] = 0
                    del self.runners[runner_id]
                    self.num_runners -= 1
                    self.global_state[_SIG][self.timestep :, runner_id] = 0

        if self.timestep == self.episode_length:
            for runner_id in self.runners:
                rew[runner_id] += self.end_of_game_reward_for_runner

        return rew

    def get_data_dictionary(self):
        """
        Create a dictionary of data to push to the device
        """
        data_dict = DataFeed()
        for feature in [_LOC_X, _LOC_Y, _SP, _DIR, _ACC]:
            data_dict.add_data(
                name=feature,
                data=self.global_state[feature][0],
                save_copy_and_apply_at_reset=True,
            )
        data_dict.add_data(
            name="agent_types",
            data=[self.agent_type[agent_id] for agent_id in range(self.num_agents)],
        )
        data_dict.add_data(
            name="num_runners", data=self.num_runners, save_copy_and_apply_at_reset=True
        )
        data_dict.add_data(
            name="num_other_agents_observed", data=self.num_other_agents_observed
        )
        data_dict.add_data(name="grid_length", data=self.grid_length)
        data_dict.add_data(
            name="edge_hit_reward_penalty",
            data=self.edge_hit_reward_penalty,
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="step_rewards",
            data=self.step_rewards,
        )
        data_dict.add_data(name="edge_hit_penalty", data=self.edge_hit_penalty)
        data_dict.add_data(name="max_speed", data=self.max_speed)
        data_dict.add_data(name="acceleration_actions", data=self.acceleration_actions)
        data_dict.add_data(name="turn_actions", data=self.turn_actions)
        data_dict.add_data(name="skill_levels", data=self.skill_levels)
        data_dict.add_data(name="use_full_observation", data=self.use_full_observation)
        data_dict.add_data(
            name="distance_margin_for_reward", data=self.distance_margin_for_reward
        )
        data_dict.add_data(
            name="tag_reward_for_tagger", data=self.tag_reward_for_tagger
        )
        data_dict.add_data(
            name="tag_penalty_for_runner", data=self.tag_penalty_for_runner
        )
        data_dict.add_data(
            name="end_of_game_reward_for_runner",
            data=self.end_of_game_reward_for_runner,
        )
        data_dict.add_data(
            name="neighbor_distances",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="neighbor_ids_sorted_by_distance",
            data=np.zeros((self.num_agents, self.num_agents - 1), dtype=np.int32),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="nearest_neighbor_ids",
            data=np.zeros(
                (self.num_agents, self.num_other_agents_observed), dtype=np.int32
            ),
            save_copy_and_apply_at_reset=True,
        )
        data_dict.add_data(
            name="runner_exits_game_after_tagged",
            data=self.runner_exits_game_after_tagged,
        )
        data_dict.add_data(
            name="still_in_the_game",
            data=self.still_in_the_game,
            save_copy_and_apply_at_reset=True,
        )
        return data_dict

    def get_tensor_dictionary(self):
        tensor_dict = DataFeed()
        return tensor_dict

    def reset(self):
        """
        Env reset().
        """
        # Reset time to the beginning
        self.timestep = 0

        # Re-initialize the global state
        self.global_state = {}
        self.set_global_state(
            key=_LOC_X, value=self.starting_location_x, t=self.timestep
        )
        self.set_global_state(
            key=_LOC_Y, value=self.starting_location_y, t=self.timestep
        )
        self.set_global_state(key=_SP, value=self.starting_speeds, t=self.timestep)
        self.set_global_state(key=_DIR, value=self.starting_directions, t=self.timestep)
        self.set_global_state(
            key=_ACC, value=self.starting_accelerations, t=self.timestep
        )

        # Array to keep track of the agents that are still in play
        self.still_in_the_game = np.ones(self.num_agents, dtype=self.int_dtype)

        # Initialize global state for "still_in_the_game" to all ones
        self.global_state[_SIG] = np.ones(
            (self.episode_length + 1, self.num_agents), dtype=self.int_dtype
        )

        # Penalty for hitting the edges
        self.edge_hit_reward_penalty = np.zeros(self.num_agents, dtype=self.float_dtype)

        # Reinitialize some variables that may have changed during previous episode
        self.runners = copy.deepcopy(self.runners_at_reset)
        self.num_runners = len(self.runners)

        return self.generate_observation()

    def step(self, actions=None):
        """
        Env step() - The GPU version calls the corresponding CUDA kernels
        """
        self.timestep += 1
        if self.use_cuda:
            # CUDA version of step()
            # This subsumes update_state(), generate_observation(),
            # and compute_reward()

            args = [
                _LOC_X,
                _LOC_Y,
                _SP,
                _DIR,
                _ACC,
                "agent_types",
                "edge_hit_reward_penalty",
                "edge_hit_penalty",
                "grid_length",
                "acceleration_actions",
                "turn_actions",
                "max_speed",
                "num_other_agents_observed",
                "skill_levels",
                "runner_exits_game_after_tagged",
                "still_in_the_game",
                "use_full_observation",
                _OBSERVATIONS,
                _ACTIONS,
                "neighbor_distances",
                "neighbor_ids_sorted_by_distance",
                "nearest_neighbor_ids",
                _REWARDS,
                "step_rewards",
                "num_runners",
                "distance_margin_for_reward",
                "tag_reward_for_tagger",
                "tag_penalty_for_runner",
                "end_of_game_reward_for_runner",
                "_done_",
                "_timestep_",
                ("n_agents", "meta"),
                ("episode_length", "meta"),
            ]
            self.cuda_step(
                *self.cuda_step_function_feed(args),
                block=self.cuda_function_manager.block,
                grid=self.cuda_function_manager.grid,
            )
            result = None  # do not return anything
        else:
            assert isinstance(actions, dict)
            assert len(actions) == self.num_agents

            acceleration_action_ids = [
                actions[agent_id][0] for agent_id in range(self.num_agents)
            ]
            turn_action_ids = [
                actions[agent_id][1] for agent_id in range(self.num_agents)
            ]

            assert all(
                0 <= acc <= self.num_acceleration_levels
                for acc in acceleration_action_ids
            )
            assert all(0 <= turn <= self.num_turn_levels for turn in turn_action_ids)

            delta_accelerations = self.acceleration_actions[acceleration_action_ids]
            delta_turns = self.turn_actions[turn_action_ids]

            # Update state and generate observation
            self.update_state(delta_accelerations, delta_turns)
            if not self.use_cuda:
                obs = self.generate_observation()

            # Compute rewards and done
            rew = self.compute_reward()
            done = {
                "__all__": (self.timestep >= self.episode_length)
                or (self.num_runners == 0)
            }
            info = {}

            result = obs, rew, done, info
        return result
