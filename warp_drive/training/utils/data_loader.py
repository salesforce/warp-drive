# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np
from gym.spaces import Discrete, MultiDiscrete

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
    env_wrapper, policy_tag_to_agent_id_map, training_batch_size_per_env
):
    """
    Create observations, sampled_actions, rewards and done flags placeholders
    and push to the device; this is required for generating environment
    roll-outs as well as training.
    """
    assert env_wrapper is not None
    assert isinstance(policy_tag_to_agent_id_map, dict)
    assert len(policy_tag_to_agent_id_map) > 0  # at least one policy

    # Reset the environment
    obs = env_wrapper.reset_all_envs()

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

    # Note: This release version assumes all agents use the same obs/action space!
    # If the obs/action spaces for the agents are different, we would just need to
    # push obs/action and reward placeholders for each agent separately.

    # Assert all action spaces are of the same type
    action_spaces = [action_space[key] for key in action_space]
    action_types = [type(action_space[key]) for key in action_space]
    assert all_equal(action_types)

    # Also assert all action spaces are of the same dimension
    first_agent_action_space = action_spaces[0]

    if isinstance(first_agent_action_space, MultiDiscrete):
        action_dims = [tuple(action_space[key].nvec) for key in action_space]
    elif isinstance(first_agent_action_space, Discrete):
        action_dims = [tuple([action_space[key].n]) for key in action_space]
    else:
        raise NotImplementedError(
            "Action spaces can be of type 'Discrete' or 'MultiDiscrete'"
        )
    assert all_equal(action_dims)

    # Use the first action dimension element's length to determine the number of actions
    num_actions = len(action_dims[0])

    # Create observations, sampled_actions and rewards placeholders
    # Note: We add the "num_envs" dimension to the placehholders since we will
    # be running multiple replicas of the environment concurrently.
    num_envs = env_wrapper.n_envs
    observations_placeholder = np.stack(
        [np.array([obs[key] for key in sorted(obs.keys())]) for _ in range(num_envs)],
        axis=0,
    )
    sampled_actions_placeholder = np.zeros(
        (num_envs, env_wrapper.env.num_agents, num_actions),
        dtype=np.int32,
    )
    rewards_placeholder = np.zeros((num_envs, env_wrapper.n_agents), dtype=np.float32)

    # Use the DataFeed class to add the observations, actions and rewards
    # placeholder arrays. These arrays will be written to during the environment step().
    # For the observations placeholders set save_copy_and_apply_at_reset=True, so that a
    # copy of the initial observation values will be saved and applied at every reset.
    tensor_feed = DataFeed()
    tensor_feed.add_data(
        name=_OBSERVATIONS,
        data=observations_placeholder,
        save_copy_and_apply_at_reset=True,
    )
    tensor_feed.add_data(name=_ACTIONS, data=sampled_actions_placeholder)
    tensor_feed.add_data(name=_REWARDS, data=rewards_placeholder)

    # Also add separate placeholders for each policy model's sampled actions,
    # if there are multiple policies or a MultiDiscrete action space.
    # This is required since our sampler will be invoked for each policy model and
    # action dimension separately.
    single_policy_with_discrete_action_space = len(
        policy_tag_to_agent_id_map
    ) == 1 and isinstance(first_agent_action_space, Discrete)
    if not single_policy_with_discrete_action_space:
        for pol_mod_tag in policy_tag_to_agent_id_map:
            if isinstance(first_agent_action_space, Discrete):
                tensor_feed.add_data(
                    name=f"{_ACTIONS}_{pol_mod_tag}_0",
                    data=sampled_actions_placeholder[
                        :, policy_tag_to_agent_id_map[pol_mod_tag]
                    ],
                )
            elif isinstance(first_agent_action_space, MultiDiscrete):
                for action_idx in range(num_actions):
                    tensor_feed.add_data(
                        name=f"{_ACTIONS}_{pol_mod_tag}_{action_idx}",
                        data=sampled_actions_placeholder[
                            :, policy_tag_to_agent_id_map[pol_mod_tag], action_idx
                        ],
                    )
            else:
                raise NotImplementedError(
                    "Action spaces can be of type 'Discrete' or 'MultiDiscrete'"
                )

    # Additionally, add placeholders for the sampled_actions, rewards and done_flags
    # for the roll-out batch.
    # Note: The observation batch will be defined by the Trainer after processing.
    # The batch of observations, sampled_actions, rewards and done flags are
    # needed for training.
    tensor_feed.add_data(
        name=f"{_ACTIONS}_batch",
        data=np.zeros(
            (training_batch_size_per_env,) + sampled_actions_placeholder.shape,
            dtype=np.int32,
        ),
    )
    tensor_feed.add_data(
        name=f"{_REWARDS}_batch",
        data=np.zeros((training_batch_size_per_env,) + rewards_placeholder.shape),
    )
    done_flags_placeholder = env_wrapper.cuda_data_manager.pull_data_from_device(
        "_done_"
    )
    tensor_feed.add_data(
        name=f"{_DONE_FLAGS}_batch",
        data=np.zeros((training_batch_size_per_env,) + done_flags_placeholder.shape),
    )
    # Push all the placeholders to the device (GPU)
    env_wrapper.cuda_data_manager.push_data_to_device(
        tensor_feed, torch_accessible=True
    )
