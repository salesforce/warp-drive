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
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_ACTION_MASK = Constants.ACTION_MASK
_REWARDS = Constants.REWARDS
_DONE_FLAGS = Constants.DONE_FLAGS


def all_equal(iterable):
    """
    Check all elements of an iterable (e.g., list) are identical
    """
    return len(set(iterable)) <= 1


def create_and_push_data_placeholders(
    env_wrapper=None,
    action_sampler=None,
    policy_tag_to_agent_id_map=None,
    create_separate_placeholders_for_each_policy=False,
    obs_dim_corresponding_to_num_agents="first",
    training_batch_size_per_env=None,
    push_data_batch_placeholders=True,
):
    """
    Create observations, sampled_actions, rewards and done flags placeholders
    and push to the device; this is required for generating environment
    roll-outs as well as training.
    :param env_wrapper: the wrapped environment object.
    :param action_sampler: the sample controller to register and sample actions
    :param policy_tag_to_agent_id_map:
        a dictionary mapping policy tag to agent ids.
    :param create_separate_placeholders_for_each_policy:
        flag indicating whether there exist separate observations,
        actions and rewards placeholders, for each policy,
        as designed in the step function. The placeholders will be
        used in the step() function and during training.
        This flag will be False when there's only a single policy, or
        when there are multiple policies, yet all the agents have the same
        obs and action space shapes, so we may share the same placeholder.
        Defaults to False.
        Note: starting from version 2.2, and this flag is not for batch data anymore.
        Training batches are always separated by policies because the training
        is always separated by policies.
    :param obs_dim_corresponding_to_num_agents:
        indicative of which dimension in the observation corresponds
        to the number of agents, as designed in the step function.
        It may be "first" or "last". In other words,
        observations may be shaped (num_agents, *feature_dim) or
        (*feature_dim, num_agents). This is required in order for
        WarpDrive to process the observations correctly. This is only
        relevant when a single obs key corresponds to multiple agents.
        Defaults to "first".
    :param training_batch_size_per_env: the training batch size for each env.
        training_batch_size_per_env = training_batch_size // num_envs
    :param push_data_batch_placeholders: an optional flag to push placeholders
        for the batches of actions, rewards and the done flags.
        Defaults to True.
    :return:
    1. create step-wise observation, action, reward placeholders

        observation placeholder:
            if create_separate_placeholders_for_each_policy:
                name: f"{_OBSERVATIONS}" + f"_{policy_name}" + f"_{obs_key}"
                shape: [num_envs, policy_tag_to_agent_id_map[policy_name], observation_size]
            else:
                name: f"{_OBSERVATIONS}" + f"_{obs_key}"
                shape: [num_envs, num_agents, observation_size]

        action placeholder:
            if create_separate_placeholders_for_each_policy:
                if MultiDiscrete:
                    - The following is used for sampler to sample each individual actions
                    name: f"{_ACTIONS}_{action_type_id}" + f"_{policy_name}"
                    shape: [num_envs, policy_tag_to_agent_id_map[policy_name], 1]
                    - The followings is used for action placeholders
                    name: f"{_ACTIONS}" + f"_{policy_name}"
                    shape: [num_envs, policy_tag_to_agent_id_map[policy_name], num_action_types]
                elif Discrete:
                    name: f"{_ACTIONS}" + f"_{policy_name}"
                    shape: [num_envs, policy_tag_to_agent_id_map[policy_name], 1]
            else:
                if MultiDiscrete:
                    - The following is used for sampler to sample each individual actions
                    name: f"{_ACTIONS}_{action_type_id}"
                    shape: [num_envs, num_agents, 1]
                    - The followings is used for action placeholders
                    name: f"{_ACTIONS}"
                    shape: [num_envs, num_agents, num_action_types]

                elif Discrete:
                    name: f"{_ACTIONS}"
                    shape: [num_envs, num_agents, 1]

        reward placeholder:
        if create_separate_placeholders_for_each_policy:
            name: f"{_REWARDS}" + f"_{policy_name}"
            shape: [num_envs, policy_tag_to_agent_id_map[policy_name]]
        else:
            name: f"{_REWARDS}"
            shape: [num_envs, num_agents]

    2. create batch action, reward and done for EACH policy
       Note: They will be created if push_data_batch_placeholders is True.
             Starting from version 2.2, training batches are always separated by policies because the training
             is always separated by policies.
        if push_data_batch_placeholders:
            name: f"{_ACTIONS/_REWARDS/_DONES}_batch" + f"_{policy_name}"
            shape: [training_batch_size_per_env, num_envs, policy_tag_to_agent_id_map[policy_name], ...]

    3. create batch processed observation for EACH policy
       Note: They will always be created as long as training_batch_size_per_env is defined
             This is the default way of the flattened observation batch to feed into model.forward()
        if training_batch_size_per_env is not None:
            name: f"{_PROCESSED_OBSERVATIONS}_batch" + f"_{policy_name}"
            shape: (training_batch_size_per_env, ) +
                   (num_envs, policy_tag_to_agent_id_map[policy_name], ) +
                   (processed_obs_size = get_flattened_obs_size(observation_space),)

    """
    assert env_wrapper is not None
    assert not env_wrapper.env_backend == "cpu"
    policy_tag_to_agent_id_map = _validate_policy_tag_to_agent_id_map(
        env_wrapper, policy_tag_to_agent_id_map
    )
    if push_data_batch_placeholders:
        assert training_batch_size_per_env > 0, \
            "push_data_batch_placeholders is True, but training_batch_size_per_env is not defined"

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
            _create_observation_placeholders_helper(
                env_wrapper,
                relevant_agent_ids,
                obs_dim_corresponding_to_num_agents,
                policy_suffix=f"_{pol_mod_tag}"
            )
            _create_action_placeholders_helper(
                env_wrapper,
                relevant_agent_ids,
                policy_suffix=f"_{pol_mod_tag}",
            )
            _create_reward_placeholders_helper(
                env_wrapper,
                relevant_agent_ids,
                policy_suffix=f"_{pol_mod_tag}",
            )
            if action_sampler:
                _prepare_action_sampler_helper(
                    env_wrapper,
                    action_sampler,
                    relevant_agent_ids,
                    policy_suffix=f"_{pol_mod_tag}",
                )
            _log_obs_action_spaces(pol_mod_tag, relevant_agent_ids[0], env_wrapper)
    else:
        # When there's only a single policy, or there are multiple policies, yet
        # all the agents have the same obs/action space,
        # this scenario will be used
        relevant_agent_ids = list(range(env_wrapper.n_agents))
        if len(relevant_agent_ids) > 1:
            _validate_obs_action_spaces(relevant_agent_ids, env_wrapper)
        _create_observation_placeholders_helper(
            env_wrapper,
            relevant_agent_ids,
            obs_dim_corresponding_to_num_agents,
        )
        _create_action_placeholders_helper(
            env_wrapper,
            relevant_agent_ids,
        )
        _create_reward_placeholders_helper(
            env_wrapper,
            relevant_agent_ids,
        )
        if action_sampler:
            _prepare_action_sampler_helper(
                env_wrapper,
                action_sampler,
                relevant_agent_ids,
            )
        for pol_mod_tag in policy_tag_to_agent_id_map:
            agent_ids = policy_tag_to_agent_id_map[pol_mod_tag]
            _log_obs_action_spaces(pol_mod_tag, agent_ids[0], env_wrapper)

    if training_batch_size_per_env is not None and training_batch_size_per_env > 1:
        for pol_mod_tag in policy_tag_to_agent_id_map:
            relevant_agent_ids = policy_tag_to_agent_id_map[pol_mod_tag]
            _create_processed_observation_batches_helper(
                env_wrapper,
                relevant_agent_ids,
                training_batch_size_per_env,
                batch_suffix=f"_{pol_mod_tag}",
            )

    # Actions, rewards and done batches are optional depending on the batching logic
    if push_data_batch_placeholders:
        for pol_mod_tag in policy_tag_to_agent_id_map:
            relevant_agent_ids = policy_tag_to_agent_id_map[pol_mod_tag]
            _create_action_batches_helper(
                env_wrapper,
                relevant_agent_ids,
                training_batch_size_per_env,
                batch_suffix=f"_{pol_mod_tag}",
            )
            _create_reward_batches_helper(
                env_wrapper,
                relevant_agent_ids,
                training_batch_size_per_env,
                batch_suffix=f"_{pol_mod_tag}",
            )
        _create_done_batches_helper(
            env_wrapper,
            training_batch_size_per_env,
        )


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


def _create_observation_placeholders_helper(
    env_wrapper,
    agent_ids,
    obs_dim_corresponding_to_num_agents,
    policy_suffix="",
):
    """
    Helper function to create obs placeholders
    """
    # Use the DataFeed class to add the observations, actions
    # and rewards placeholder arrays.
    # These arrays will be written to during the environment step().
    tensor_feed = DataFeed()

    num_envs = env_wrapper.n_envs
    obs = [env_wrapper.obs_at_reset() for _ in range(num_envs)]
    assert len(obs) == num_envs
    first_env_id = 0
    first_agent_id = agent_ids[0]

    if isinstance(obs[first_env_id][first_agent_id], (list, np.ndarray)):
        agent_obs_for_all_envs = [
            get_obs(obs[env_id], agent_ids, obs_dim_corresponding_to_num_agents)
            for env_id in range(num_envs)
        ]
        stacked_obs = np.stack(agent_obs_for_all_envs, axis=0)

        tensor_feed.add_data(
            name=f"{_OBSERVATIONS}" + policy_suffix,
            data=stacked_obs,
            save_copy_and_apply_at_reset=True,
        )
    elif isinstance(obs[first_env_id][first_agent_id], dict):
        for obs_key in obs[first_env_id][first_agent_id]:
            agent_obs_for_all_envs = [
                get_obs(
                    obs[env_id], agent_ids, obs_dim_corresponding_to_num_agents, obs_key=obs_key
                )
                for env_id in range(num_envs)
            ]
            stacked_obs = np.stack(agent_obs_for_all_envs, axis=0)

            tensor_feed.add_data(
                name=f"{_OBSERVATIONS}" + policy_suffix + f"_{obs_key}",
                data=stacked_obs,
                save_copy_and_apply_at_reset=True,
            )
    else:
        raise NotImplementedError("Only array or dict type observations are supported!")

    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def _create_processed_observation_batches_helper(
    env_wrapper,
    agent_ids,
    training_batch_size_per_env,
    batch_suffix="",
    ):

    tensor_feed = DataFeed()

    num_envs = env_wrapper.n_envs
    num_agents = len(agent_ids)
    first_agent_id = agent_ids[0]

    name = f"{_PROCESSED_OBSERVATIONS}_batch" + batch_suffix
    if not env_wrapper.cuda_data_manager.is_data_on_device_via_torch(name):
        observation_space = env_wrapper.env.observation_space[first_agent_id]
        processed_obs_size = get_flattened_obs_size(observation_space)
        processed_obs_batch = np.zeros(
            (training_batch_size_per_env,)
            + (
                num_envs,
                num_agents,
            )
            + (processed_obs_size,)
            )
        tensor_feed.add_data(name=name, data=processed_obs_batch)

    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def _create_action_placeholders_helper(
    env_wrapper,
    agent_ids,
    policy_suffix="",
    ):

    tensor_feed = DataFeed()

    num_envs = env_wrapper.n_envs
    num_agents = len(agent_ids)
    first_agent_id = agent_ids[0]
    action_space = env_wrapper.env.action_space[first_agent_id]

    if isinstance(action_space, Discrete):
        tensor_feed.add_data(
            name=_ACTIONS + policy_suffix,
            data=np.zeros(
                (num_envs,  num_agents, 1),
                dtype=np.int32,
            ),
        )

    elif isinstance(action_space, MultiDiscrete):
        action_dim = action_space.nvec
        num_action_types = len(action_dim)
        # Add separate placeholders for a MultiDiscrete action space.
        # This is required since our sampler will be invoked for each
        # action dimension separately.
        assert num_action_types > 1
        for action_type_id in range(num_action_types):
            tensor_feed.add_data(
                name=f"{_ACTIONS}_{action_type_id}" + policy_suffix,
                data=np.zeros(
                    (num_envs, num_agents, 1),
                    dtype=np.int32,
                ),
            )
        tensor_feed.add_data(
            name=_ACTIONS + policy_suffix,
            data=np.zeros(
                (
                    num_envs,
                    num_agents,
                )
                + (num_action_types,),
                dtype=np.int32,
            ),
        )

    else:
        raise NotImplementedError(
            "Only 'Discrete' or 'MultiDiscrete' type action spaces are supported!"
        )

    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def _create_action_batches_helper(
    env_wrapper,
    agent_ids,
    training_batch_size_per_env,
    batch_suffix="",
    ):

    tensor_feed = DataFeed()

    num_envs = env_wrapper.n_envs
    num_agents = len(agent_ids)
    first_agent_id = agent_ids[0]
    action_space = env_wrapper.env.action_space[first_agent_id]
    if isinstance(action_space, MultiDiscrete):
        action_dim = action_space.nvec
        num_action_types = len(action_dim)
    else:
        num_action_types = 1

    name = f"{_ACTIONS}_batch" + batch_suffix
    if not env_wrapper.cuda_data_manager.is_data_on_device(name):
        tensor_feed.add_data(
            name=name,
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

    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def _prepare_action_sampler_helper(
    env_wrapper,
    action_sampler,
    agent_ids,
    policy_suffix="",
    ):

    first_agent_id = agent_ids[0]
    action_space = env_wrapper.env.action_space[first_agent_id]

    if isinstance(action_space, Discrete):
        action_dim = action_space.n
        action_sampler.register_actions(
            env_wrapper.cuda_data_manager,
            action_name=_ACTIONS + policy_suffix,
            num_actions=action_dim,
        )
    elif isinstance(action_space, MultiDiscrete):
        action_dim = action_space.nvec
        assert len(action_dim) > 1
        for action_type_id, action_type_dim in enumerate(action_dim):
            action_sampler.register_actions(
                env_wrapper.cuda_data_manager,
                action_name=f"{_ACTIONS}_{action_type_id}" + policy_suffix,
                num_actions=action_type_dim,
            )
    else:
        raise NotImplementedError(
            "Only 'Discrete' or 'MultiDiscrete' type action spaces are supported!"
        )


def _create_reward_placeholders_helper(
    env_wrapper,
    agent_ids,
    policy_suffix="",
    ):

    tensor_feed = DataFeed()
    num_envs = env_wrapper.n_envs
    num_agents = len(agent_ids)

    # Push rewards to the device
    rewards_placeholder = np.zeros((num_envs, num_agents), dtype=np.float32)
    tensor_feed.add_data(name=_REWARDS + policy_suffix, data=rewards_placeholder)

    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def _create_reward_batches_helper(
    env_wrapper,
    agent_ids,
    training_batch_size_per_env,
    batch_suffix="",
    ):

    tensor_feed = DataFeed()

    num_envs = env_wrapper.n_envs
    num_agents = len(agent_ids)

    name = f"{_REWARDS}_batch" + batch_suffix
    if not env_wrapper.cuda_data_manager.is_data_on_device(name):
        tensor_feed.add_data(
            name=name,
            data=np.zeros(
                (training_batch_size_per_env,)
                + (
                    num_envs,
                    num_agents,
                ),
                dtype=np.float32,
                ),
        )

    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def _create_done_batches_helper(
    env_wrapper,
    training_batch_size_per_env,
    ):

    tensor_feed = DataFeed()
    name = f"{_DONE_FLAGS}_batch"
    if not env_wrapper.cuda_data_manager.is_data_on_device(name):
        done_flags_placeholder_shape = (
            env_wrapper.cuda_data_manager.get_shape("_done_")
        )
        tensor_feed.add_data(
            name=name,
            data=np.zeros(
                (training_batch_size_per_env,) + done_flags_placeholder_shape,
                dtype=np.int32,
            ),
        )

    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )


def get_obs(obs, agent_ids, obs_dim_corresponding_to_num_agents="first", obs_key=None):
    if obs_key is not None:
        agent_obs = np.array([obs[agent_id][obs_key] for agent_id in agent_ids])
    else:
        agent_obs = np.array([obs[agent_id] for agent_id in agent_ids])

    if obs_dim_corresponding_to_num_agents == "last" and len(agent_ids) > 1:
        return np.swapaxes(agent_obs, 0, -1)

    return agent_obs


def get_flattened_obs_size(observation_space):
    """
    Get the total size of the observations after flattening.
    This shall be the default flattening strategy for processed observations
    """
    if isinstance(observation_space, Box):
        obs_size = np.prod(observation_space.shape)
    elif isinstance(observation_space, Dict):
        obs_size = 0
        for obs_key in observation_space:
            if obs_key == _ACTION_MASK:
                pass
            else:
                obs_size += np.prod(observation_space[obs_key].shape)
    else:
        raise NotImplementedError("Observation space must be of Box or Dict type")
    return int(obs_size)
