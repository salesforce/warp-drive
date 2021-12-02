# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from gym.spaces import Box, Dict, Discrete, MultiDiscrete

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS
_DONE_FLAGS = Constants.DONE_FLAGS


def all_equal(iterable):
    """
    Check all elements of an iterable (e.g., list) are identical
    """
    return len(set(iterable)) <= 1


def create_and_push_data_placeholders(
    env_wrapper,
    policy_tag_to_agent_id_map,
    training_batch_size_per_env,
    create_separate_placeholders_for_each_policy,
):
    """
    Create observations, sampled_actions, rewards and done flags placeholders
    and push to the device; this is required for generating environment
    roll-outs as well as training.
    """
    assert env_wrapper is not None
    assert isinstance(policy_tag_to_agent_id_map, dict)
    assert len(policy_tag_to_agent_id_map) > 0  # at least one policy

    observation_space = {}
    action_space = {}
    # Define the action spaces for each policy
    for pol_mod_tag in policy_tag_to_agent_id_map:
        first_agent_id = policy_tag_to_agent_id_map[pol_mod_tag][0]

        observation_space[pol_mod_tag] = env_wrapper.env.observation_space[
            first_agent_id
        ]
        action_space[pol_mod_tag] = env_wrapper.env.action_space[first_agent_id]
        print("-" * 40)
        print(f"Observation space: {pol_mod_tag}", observation_space[pol_mod_tag])
        print(f"Action space: {pol_mod_tag}", action_space[pol_mod_tag])
        print("-" * 40)

    # Reset the environment
    obs = env_wrapper.reset_all_envs()
    num_envs = env_wrapper.n_envs

    tensor_feed = DataFeed()
    # Use the DataFeed class to add the observations, actions
    # and rewards placeholder arrays.
    # These arrays will be written to during the environment step().
    if create_separate_placeholders_for_each_policy:
        # This scenario requires that we have multiple policy tags.
        # Also, if the obs/action spaces for the agents are different,
        # we will need to use this scenario to push obs actions and
        # rewards placeholders for each policy separately.

        assert len(policy_tag_to_agent_id_map) > 1
        # Create separate observations, sampled_actions and rewards placeholders
        # for each policy.
        for pol_mod_tag in policy_tag_to_agent_id_map:
            assert pol_mod_tag in obs

            # For the observations placeholders set save_copy_and_apply_at_reset=True,
            # so that a copy of the initial observation values will be saved and
            # applied at every reset.
            # Also note that we add the "num_envs" dimension to the placeholders since
            # we will be running multiple replicas of the environment concurrently.
            if isinstance(obs[pol_mod_tag], (list, np.ndarray)):
                observations_placeholder = np.stack(
                    [obs[pol_mod_tag] for _ in range(num_envs)], axis=0
                )
                tensor_feed.add_data(
                    name=f"{_OBSERVATIONS}_{pol_mod_tag}",
                    data=observations_placeholder,
                    save_copy_and_apply_at_reset=True,
                )
            elif isinstance(obs[pol_mod_tag], dict):
                for key in obs[pol_mod_tag]:
                    observations_placeholder = np.stack(
                        [obs[pol_mod_tag][key] for _ in range(num_envs)], axis=0
                    )
                    tensor_feed.add_data(
                        name=f"{_OBSERVATIONS}_{pol_mod_tag}_{key}",
                        data=observations_placeholder,
                        save_copy_and_apply_at_reset=True,
                    )
            else:
                raise NotImplementedError("obs may be an array-type or a dictionary")

            tensor_feed = create_and_push_data_placeholders_helper(
                num_envs,
                len(policy_tag_to_agent_id_map[pol_mod_tag]),
                training_batch_size_per_env,
                action_space[pol_mod_tag],
                tensor_feed,
                suffix=f"_{pol_mod_tag}",
            )
    else:
        # When there's only a single policy, this scenario will be used!
        # It can also be used only when there are multiple policies, yet
        # all the agents have the same obs/action space!

        # Assert all observation spaces are of the same type
        observation_types = [
            type(obs_space) for obs_space in observation_space.values()
        ]
        assert all_equal(observation_types)

        # Also assert all observation spaces are of the same shape
        first_agent_observation_space = list(observation_space.values())[0]

        if isinstance(first_agent_observation_space, Box):
            observation_shapes = [
                obs_space.shape for obs_space in observation_space.values()
            ]
            assert all_equal(observation_shapes)
        elif isinstance(first_agent_observation_space, Dict):
            observation_shape_keys = [
                tuple([obs_space.spaces.keys])
                for obs_space in observation_space.values()
            ]
            assert all_equal(observation_shape_keys)
            observation_shape_values = [
                tuple([obs_space.spaces.values])
                for obs_space in observation_space.values()
            ]
            assert all_equal(observation_shape_values)
        else:
            raise NotImplementedError(
                "Observation spaces can be of type 'Box' or 'Dict'"
            )

        # Assert all action spaces are of the same type
        action_types = [type(act_space) for act_space in action_space.values()]
        assert all_equal(action_types)

        # Also assert all action spaces are of the same dimension
        first_agent_action_space = list(action_space.values())[0]

        if isinstance(first_agent_action_space, MultiDiscrete):
            action_dims = [tuple(act_space.nvec) for act_space in action_space.values()]
        elif isinstance(first_agent_action_space, Discrete):
            action_dims = [tuple([act_space.n]) for act_space in action_space.values()]
        else:
            raise NotImplementedError(
                "Action spaces can be of type 'Discrete' or 'MultiDiscrete'"
            )
        assert all_equal(action_dims)

        # Observations
        first_policy = list(observation_space.keys())[0]
        first_agent_id = policy_tag_to_agent_id_map[first_policy][0]

        if isinstance(obs[first_agent_id], (list, np.ndarray)):
            observations_placeholder = np.stack(
                [
                    np.array([obs[agent_id] for agent_id in sorted(obs.keys())])
                    for _ in range(num_envs)
                ],
                axis=0,
            )
            tensor_feed.add_data(
                name=f"{_OBSERVATIONS}",
                data=observations_placeholder,
                save_copy_and_apply_at_reset=True,
            )
        elif isinstance(obs[first_agent_id], dict):
            for key in obs[first_agent_id]:
                observations_placeholder = np.stack(
                    [
                        np.array(
                            [obs[agent_id][key] for agent_id in sorted(obs.keys())]
                        )
                        for _ in range(num_envs)
                    ],
                    axis=0,
                )
                tensor_feed.add_data(
                    name=f"{_OBSERVATIONS}_{key}",
                    data=observations_placeholder,
                    save_copy_and_apply_at_reset=True,
                )
        else:
            raise NotImplementedError("obs may be an array-type or a dictionary")

        tensor_feed = create_and_push_data_placeholders_helper(
            num_envs,
            env_wrapper.n_agents,
            training_batch_size_per_env,
            first_agent_action_space,
            tensor_feed,
        )

    # Done flags placeholders for the roll-out batch
    done_flags_placeholder = env_wrapper.cuda_data_manager.pull_data_from_device(
        "_done_"
    )
    tensor_feed.add_data(
        name=f"{_DONE_FLAGS}_batch",
        data=np.zeros(
            (training_batch_size_per_env,) + done_flags_placeholder.shape,
            dtype=np.int32,
        ),
    )

    # Push all the placeholders to the device (GPU)
    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def create_and_push_data_placeholders_helper(
    num_envs,
    num_agents,
    training_batch_size_per_env,
    action_space,
    tensor_feed,
    suffix="",
):
    # Helper function to create and push actions and rewards placeholders

    # Sampled actions
    if isinstance(action_space, MultiDiscrete):
        action_dim = action_space.nvec
    elif isinstance(action_space, Discrete):
        action_dim = [action_space.n]
    else:
        raise NotImplementedError(
            "Action spaces can be of type 'Discrete' or 'MultiDiscrete'"
        )

    num_action_types = len(action_dim)

    if isinstance(action_space, Discrete):
        assert num_action_types == 1
        sampled_actions_placeholder = np.zeros(
            (num_envs, num_agents, num_action_types),
            dtype=np.int32,
        )
        tensor_feed.add_data(name=_ACTIONS + suffix, data=sampled_actions_placeholder)
    elif isinstance(action_space, MultiDiscrete):
        # Add separate placeholders for a MultiDiscrete action space.
        # This is required since our sampler will be invoked for each
        # action dimension separately.

        sampled_actions_placeholder = np.zeros(
            (num_envs, num_agents),
            dtype=np.int32,
        )
        assert num_action_types > 1
        for action_idx in range(num_action_types):
            tensor_feed.add_data(
                name=f"{_ACTIONS}_{action_idx}" + suffix,
                data=sampled_actions_placeholder,
            )
        tensor_feed.add_data(
            name=_ACTIONS + suffix,
            data=np.zeros(
                sampled_actions_placeholder.shape + (num_action_types,),
                dtype=np.int32,
            ),
        )
    else:
        raise NotImplementedError(
            "Action spaces can be of type 'Discrete' or 'MultiDiscrete'"
        )

    tensor_feed.add_data(
        name=f"{_ACTIONS}_batch" + suffix,
        data=np.zeros(
            (training_batch_size_per_env,)
            + (
                num_envs,
                num_agents,
            )
            + (num_action_types,),
            dtype=np.int32,
        ),
    )

    # Rewards
    rewards_placeholder = np.zeros((num_envs, num_agents), dtype=np.float32)
    tensor_feed.add_data(name=_REWARDS + suffix, data=rewards_placeholder)
    tensor_feed.add_data(
        name=f"{_REWARDS}_batch" + suffix,
        data=np.zeros((training_batch_size_per_env,) + rewards_placeholder.shape),
    )

    return tensor_feed
