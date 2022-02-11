# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging

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
    env_wrapper=None,
    policy_tag_to_agent_id_map=None,
    create_separate_placeholders_for_each_policy=False,
    obs_dim_corresponding_to_num_agents="first",
    training_batch_size_per_env=1,
    push_data_batch_placeholders=True,
):
    """
    Create observations, sampled_actions, rewards and done flags placeholders
    and push to the device; this is required for generating environment
    roll-outs as well as training.
    env_wrapper: the wrapped environment object.
    policy_tag_to_agent_id_map:
        a dictionary mapping policy tag to agent ids.
    create_separate_placeholders_for_each_policy:
        flag indicating whether there exist separate observations,
        actions and rewards placeholders, for each policy,
        as designed in the step function. The placeholders will be
        used in the step() function and during training.
        When there's only a single policy, this flag will be False.
        It can also be True when there are multiple policies, yet
        all the agents have the same obs and action space shapes,
        so we can share the same placeholder.
        Defaults to False.
    obs_dim_corresponding_to_num_agents:
        indicative of which dimension in the observation corresponds
        to the number of agents, as designed in the step function.
        It may be "first" or "last". In other words,
        observations may be shaped (num_agents, *feature_dim) or
        (*feature_dim, num_agents). This is required in order for
        WarpDrive to process the observations correctly. This is only
        relevant when a single obs key corresponds to multiple agents.
        Defaults to "first".
    training_batch_size_per_env: the training batch size for each env.
    push_data_batch_placeholders: an optional flag to push placeholders
        for the batches of actions, rewards and the done flags.
        Defaults to True.
    """
    assert env_wrapper is not None
    assert env_wrapper.use_cuda
    policy_tag_to_agent_id_map = _validate_policy_tag_to_agent_id_map(
        env_wrapper, policy_tag_to_agent_id_map
    )
    assert training_batch_size_per_env > 0

    # Assert all the relevant agents' obs and action spaces are identical
    if create_separate_placeholders_for_each_policy:
        # This scenario requires that we have multiple policy tags.
        # Also, if the obs/action spaces for the agents are different,
        # we will need to use this scenario to push obs actions and
        # rewards placeholders for each policy separately.
        assert len(policy_tag_to_agent_id_map) > 1
        for pol_mod_tag in policy_tag_to_agent_id_map:
            relevant_agent_ids = policy_tag_to_agent_id_map[pol_mod_tag]
            if len(relevant_agent_ids) > 1:
                _validate_obs_action_spaces(relevant_agent_ids, env_wrapper)
            # Create separate observations, sampled_actions and rewards placeholders
            # for each policy.
            _create_and_push_data_placeholders_helper(
                env_wrapper,
                policy_tag_to_agent_id_map,
                obs_dim_corresponding_to_num_agents,
                training_batch_size_per_env,
                push_data_batch_placeholders,
                suffix=f"_{pol_mod_tag}",
            )
            _log_obs_action_spaces(pol_mod_tag, relevant_agent_ids[0], env_wrapper)
    else:
        # When there's only a single policy, this scenario will be used!
        # It can also be used only when there are multiple policies, yet
        # all the agents have the same obs/action space!
        relevant_agent_ids = list(range(env_wrapper.n_agents))
        if len(relevant_agent_ids) > 1:
            _validate_obs_action_spaces(relevant_agent_ids, env_wrapper)
        _create_and_push_data_placeholders_helper(
            env_wrapper,
            policy_tag_to_agent_id_map,
            obs_dim_corresponding_to_num_agents,
            training_batch_size_per_env,
            push_data_batch_placeholders,
        )
        for pol_mod_tag in policy_tag_to_agent_id_map:
            agent_ids = policy_tag_to_agent_id_map[pol_mod_tag]
            _log_obs_action_spaces(pol_mod_tag, agent_ids[0], env_wrapper)


def _validate_policy_tag_to_agent_id_map(env_wrapper, policy_tag_to_agent_id_map):
    # Reset the environment to obtain the obs dict
    obs = env_wrapper.obs_at_reset()
    if policy_tag_to_agent_id_map is None:
        logging.info("Using a shared policy across all agents.")
        policy_tag_to_agent_id_map = {"shared": sorted(list(obs.keys()))}
    assert isinstance(policy_tag_to_agent_id_map, dict)
    assert len(policy_tag_to_agent_id_map) > 0  # at least one policy
    for key in policy_tag_to_agent_id_map:
        assert isinstance(policy_tag_to_agent_id_map[key], (list, tuple, np.ndarray))
    # Ensure that each agent is only mapped to one policy
    agent_id_to_policy_map = {}
    for pol_tag, agent_ids in policy_tag_to_agent_id_map.items():
        for agent_id in agent_ids:
            assert (
                agent_id not in agent_id_to_policy_map
            ), f"{agent_id} is mapped to multiple policies!"
            agent_id_to_policy_map[agent_id] = pol_tag
    # Ensure that every agent is mapped to a policy
    for agent_id in obs:
        assert (
            agent_id in agent_id_to_policy_map
        ), f"{agent_id} is not mapped to any policy!"
    return policy_tag_to_agent_id_map


def _validate_obs_action_spaces(agent_ids, env_wrapper):
    """
    Validate that the observation and action spaces for all the agent ids
    are of the same type and have the same shapes
    """
    # Assert all the relevant agents' obs spaces are identical
    obs_spaces = [env_wrapper.env.observation_space[agent_id] for agent_id in agent_ids]
    observation_types = [type(obs_space) for obs_space in obs_spaces]
    assert all_equal(observation_types)
    first_agent_obs_space = obs_spaces[0]
    if isinstance(first_agent_obs_space, Box):
        observation_shapes = tuple(obs_space.shape for obs_space in obs_spaces)
        assert all_equal(observation_shapes)
    elif isinstance(first_agent_obs_space, Dict):
        observation_keys = [tuple(obs_space.keys()) for obs_space in obs_spaces]
        assert all_equal(observation_keys)
        observation_value_shapes = [
            tuple(val.shape for val in obs_space.values()) for obs_space in obs_spaces
        ]
        assert all_equal(observation_value_shapes)
    else:
        raise NotImplementedError(
            "Only 'Box' or 'Dict' type observation spaces are supported!"
        )
    # Assert all the relevant agents' action spaces are identical
    action_spaces = [env_wrapper.env.action_space[agent_id] for agent_id in agent_ids]
    action_types = [type(action_space) for action_space in action_spaces]
    assert all_equal(action_types)
    first_agent_action_space = action_spaces[0]
    if isinstance(first_agent_action_space, MultiDiscrete):
        action_dims = [tuple(act_space.nvec) for act_space in action_spaces]
    elif isinstance(first_agent_action_space, Discrete):
        action_dims = [tuple([act_space.n]) for act_space in action_spaces]
    else:
        raise NotImplementedError(
            "Only 'Discrete' or 'MultiDiscrete' type action spaces are supported!"
        )
    assert all_equal(action_dims)


def _log_obs_action_spaces(pol_mod_tag, agent_id, env_wrapper):
    observation_space = env_wrapper.env.observation_space[agent_id]
    action_space = env_wrapper.env.action_space[agent_id]
    logging.info("-" * 40)
    if isinstance(observation_space, Box):
        logging.info(
            f"Observation space shape: {pol_mod_tag} - {observation_space.shape}"
        )
    elif isinstance(observation_space, Dict):
        for key in observation_space:
            logging.info(
                f"Observation space shape ({key}): "
                f"{pol_mod_tag} - {observation_space[key].shape}"
            )
    else:
        raise NotImplementedError(
            "Only 'Box' or 'Dict' type observation spaces are supported!"
        )
    logging.info(f"Action space: {pol_mod_tag} - {action_space}")
    logging.info("-" * 40)


def _create_and_push_data_placeholders_helper(
    env_wrapper,
    policy_tag_to_agent_id_map,
    obs_dim_corresponding_to_num_agents,
    training_batch_size_per_env,
    push_data_batch_placeholders,
    suffix="",
):
    """
    Helper function to create and push obs, actions rewards
    and done flags placeholders
    """
    # Use the DataFeed class to add the observations, actions
    # and rewards placeholder arrays.
    # These arrays will be written to during the environment step().
    tensor_feed = DataFeed()

    num_envs = env_wrapper.n_envs
    obs = [env_wrapper.obs_at_reset() for _ in range(num_envs)]
    assert len(obs) == num_envs
    first_env_id = 0

    # Push observations to the device
    if suffix == "":
        agent_ids = sorted(list(obs[0].keys()))
        first_agent_id = agent_ids[0]
    else:
        pol_tag = suffix.split("_")[1]
        agent_ids = policy_tag_to_agent_id_map[pol_tag]
        first_agent_id = agent_ids[0]

    if isinstance(obs[first_env_id][first_agent_id], (list, np.ndarray)):
        agent_obs_for_all_envs = [
            get_obs(obs[env_id], agent_ids, obs_dim_corresponding_to_num_agents)
            for env_id in range(num_envs)
        ]
        stacked_obs = np.stack(agent_obs_for_all_envs, axis=0)

        tensor_feed.add_data(
            name=f"{_OBSERVATIONS}" + suffix,
            data=stacked_obs,
            save_copy_and_apply_at_reset=True,
        )
    elif isinstance(obs[first_env_id][first_agent_id], dict):
        for key in obs[first_env_id][first_agent_id]:
            agent_obs_for_all_envs = [
                get_obs(
                    obs[env_id], agent_ids, obs_dim_corresponding_to_num_agents, key=key
                )
                for env_id in range(num_envs)
            ]
            stacked_obs = np.stack(agent_obs_for_all_envs, axis=0)

            tensor_feed.add_data(
                name=f"{_OBSERVATIONS}" + suffix + f"_{key}",
                data=stacked_obs,
                save_copy_and_apply_at_reset=True,
            )
    else:
        raise NotImplementedError("Only array or dict type observations are supported!")

    # Push sampled actions to the device
    action_space = env_wrapper.env.action_space[first_agent_id]
    if isinstance(action_space, MultiDiscrete):
        action_dim = action_space.nvec
    elif isinstance(action_space, Discrete):
        action_dim = [action_space.n]
    else:
        raise NotImplementedError(
            "Only 'Discrete' or 'MultiDiscrete' type action spaces are supported!"
        )

    num_action_types = len(action_dim)
    num_agents = len(agent_ids)

    sampled_actions_placeholder = np.zeros(
        (num_envs, num_agents),
        dtype=np.int32,
    )
    if isinstance(action_space, Discrete):
        assert num_action_types == 1
        tensor_feed.add_data(name=_ACTIONS + suffix, data=sampled_actions_placeholder)
    elif isinstance(action_space, MultiDiscrete):
        # Add separate placeholders for a MultiDiscrete action space.
        # This is required since our sampler will be invoked for each
        # action dimension separately.
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

    if push_data_batch_placeholders:
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

    # Push rewards to the device
    rewards_placeholder = np.zeros((num_envs, num_agents), dtype=np.float32)
    tensor_feed.add_data(name=_REWARDS + suffix, data=rewards_placeholder)
    if push_data_batch_placeholders:
        tensor_feed.add_data(
            name=f"{_REWARDS}_batch" + suffix,
            data=np.zeros((training_batch_size_per_env,) + rewards_placeholder.shape),
        )

    if push_data_batch_placeholders:
        # Push done flags placeholders for the roll-out batch to the device
        # (if not already pushed)
        name = f"{_DONE_FLAGS}_batch"
        if not env_wrapper.cuda_data_manager.is_data_on_device(name):
            done_flags_placeholder = (
                env_wrapper.cuda_data_manager.pull_data_from_device("_done_")
            )
            tensor_feed.add_data(
                name=name,
                data=np.zeros(
                    (training_batch_size_per_env,) + done_flags_placeholder.shape,
                    dtype=np.int32,
                ),
            )

    # Push all the placeholders to the device (GPU)
    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def get_obs(obs, agent_ids, obs_dim_corresponding_to_num_agents="first", key=None):
    if key is not None:
        agent_obs = np.array([obs[agent_id][key] for agent_id in agent_ids])
    else:
        agent_obs = np.array([obs[agent_id] for agent_id in agent_ids])

    if obs_dim_corresponding_to_num_agents == "last" and len(agent_ids) > 1:
        return np.swapaxes(agent_obs, 0, -1)

    return agent_obs
