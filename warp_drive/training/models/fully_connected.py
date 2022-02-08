# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Fully Connected Network class
"""

import numpy as np
import torch
import torch.nn.functional as func
from gym.spaces import Box, Dict, Discrete, MultiDiscrete
from torch import nn

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

_OBSERVATIONS = Constants.OBSERVATIONS
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_ACTION_MASK = Constants.ACTION_MASK

_LARGE_NEG_NUM = -1e20


def apply_logit_mask(logits, mask=None):
    """
    Mask values of 1 are valid actions.
    Add huge negative values to logits with 0 mask values.
    """
    if mask is None:
        return logits

    logit_mask = torch.ones_like(logits) * _LARGE_NEG_NUM
    logit_mask = logit_mask * (1 - mask)
    return logits + logit_mask


# Policy networks
# ---------------
class FullyConnected(nn.Module):
    """
    Fully connected network implementation in Pytorch
    """

    name = "torch_fully_connected"

    def __init__(
        self,
        env,
        fc_dims,
        policy,
        policy_tag_to_agent_id_map,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
    ):
        super().__init__()

        self.env = env
        assert isinstance(fc_dims, list)
        num_fc_layers = len(fc_dims)
        self.policy = policy
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.create_separate_placeholders_for_each_policy = (
            create_separate_placeholders_for_each_policy
        )
        assert obs_dim_corresponding_to_num_agents in ["first", "last"]
        self.obs_dim_corresponding_to_num_agents = obs_dim_corresponding_to_num_agents

        sample_agent_id = self.policy_tag_to_agent_id_map[self.policy][0]
        # Flatten obs space
        self.observation_space = self.env.env.observation_space[sample_agent_id]
        flattened_obs_size = self.get_flattened_obs_size()

        if isinstance(self.env.env.action_space[sample_agent_id], Discrete):
            action_space = [self.env.env.action_space[sample_agent_id].n]
        elif isinstance(self.env.env.action_space[sample_agent_id], MultiDiscrete):
            action_space = self.env.env.action_space[sample_agent_id].nvec
        else:
            raise NotImplementedError

        input_dims = [flattened_obs_size] + fc_dims[:-1]
        output_dims = fc_dims

        self.fc = nn.ModuleDict()
        for fc_layer in range(num_fc_layers):
            self.fc[str(fc_layer)] = nn.Sequential(
                nn.Linear(input_dims[fc_layer], output_dims[fc_layer]),
                nn.ReLU(),
            )

        # policy network (list of heads)
        policy_heads = [None for _ in range(len(action_space))]
        self.output_dims = []  # Network output dimension(s)
        for idx, act_space in enumerate(action_space):
            self.output_dims += [act_space]
            policy_heads[idx] = nn.Linear(fc_dims[-1], act_space)
        self.policy_head = nn.ModuleList(policy_heads)

        # value-function network head
        self.vf_head = nn.Linear(fc_dims[-1], 1)

        # used for action masking
        self.action_mask = None

    def get_flattened_obs_size(self):
        """Get the total size of the observations after flattening"""
        if isinstance(self.observation_space, Box):
            obs_size = np.prod(self.observation_space.shape)
        elif isinstance(self.observation_space, Dict):
            obs_size = 0
            for key in self.observation_space:
                if key == _ACTION_MASK:
                    pass
                else:
                    obs_size += np.prod(self.observation_space[key].shape)
        else:
            raise NotImplementedError("Observation space must be of Box or Dict type")
        return int(obs_size)

    def reshape_and_flatten_obs(self, obs):
        """
        # Note: WarpDrive assumes that all the observation are shaped
        # (num_agents, *feature_dim), i.e., the observation dimension
        # corresponding to 'num_agents' is the first one. If the observation
        # dimension corresponding to num_agents is last, we will need to
        # permute the axes to align with WarpDrive's assumption.
        """
        num_envs = obs.shape[0]
        if self.create_separate_placeholders_for_each_policy:
            num_agents = len(self.policy_tag_to_agent_id_map[self.policy])
        else:
            num_agents = self.env.n_agents

        if self.obs_dim_corresponding_to_num_agents == "first":
            pass
        elif self.obs_dim_corresponding_to_num_agents == "last":
            shape_len = len(obs.shape)
            if shape_len == 1:
                obs = obs.reshape(-1, num_agents)  # valid only when num_agents = 1
            obs = obs.permute(0, -1, *range(1, shape_len - 1))
        else:
            raise ValueError(
                "num_agents can only be the first "
                "or the last dimension in the observations."
            )
        return obs.reshape(num_envs, num_agents, -1)

    def get_flattened_obs(self):
        """
        If the obs is of Box type, it will already be flattened.
        If the obs is of Dict type, then concatenate all the
        obs values and flatten them out.
        Returns the concatenated and flattened obs.

        """
        if isinstance(self.observation_space, Box):
            if self.create_separate_placeholders_for_each_policy:
                obs = self.env.cuda_data_manager.data_on_device_via_torch(
                    f"{_OBSERVATIONS}_{self.policy}"
                )
            else:
                obs = self.env.cuda_data_manager.data_on_device_via_torch(_OBSERVATIONS)

            flattened_obs = self.reshape_and_flatten_obs(obs)
        elif isinstance(self.observation_space, Dict):
            obs_dict = {}
            for key in self.observation_space:
                if self.create_separate_placeholders_for_each_policy:
                    obs = self.env.cuda_data_manager.data_on_device_via_torch(
                        f"{_OBSERVATIONS}_{self.policy}_{key}"
                    )
                else:
                    obs = self.env.cuda_data_manager.data_on_device_via_torch(
                        f"{_OBSERVATIONS}_{key}"
                    )

                if key == _ACTION_MASK:
                    self.action_mask = self.reshape_and_flatten_obs(obs)
                    assert self.action_mask.shape[-1] == sum(self.output_dims)
                else:
                    obs_dict[key] = obs

            flattened_obs_dict = {}
            for key, value in obs_dict.items():
                flattened_obs_dict[key] = self.reshape_and_flatten_obs(value)
            flattened_obs = torch.cat(list(flattened_obs_dict.values()), dim=-1)
        else:
            raise NotImplementedError("Observation space must be of Box or Dict type")
        return flattened_obs

    def forward(self, obs=None, batch_index=None, batch_size=None):
        """
        Forward pass through the model.
        Returns action probabilities and value functions.
        """
        if obs is None:
            assert batch_index < batch_size
            # Read in observation from the placeholders and flatten them
            # before passing through the fully connected layers.
            # This is particularly relevant if the observations space is a Dict.
            obs = self.get_flattened_obs()

            if self.create_separate_placeholders_for_each_policy:
                ip = obs
            else:
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[self.policy]
                ip = obs[:, agent_ids_for_policy]

            # Push the processed (in this case, flattened) obs to the GPU (device).
            # The writing happens to a specific batch index in the processed obs batch.
            # The processed obs batch is required for training.
            self.push_processed_obs_to_batch(batch_index, batch_size, ip)

        else:
            ip = obs

        # Feed through the FC layers
        for layer in range(len(self.fc)):
            op = self.fc[str(layer)](ip)
            ip = op
        logits = op

        # Compute the action probabilities and the value function estimate
        # Apply action mask to the logits as well.
        action_masks = [None for _ in range(len(self.output_dims))]
        if self.action_mask is not None:
            start = 0
            for idx, dim in enumerate(self.output_dims):
                action_masks[idx] = self.action_mask[..., start : start + dim]
                start = start + dim
        action_probs = [
            func.softmax(apply_logit_mask(ph(logits), action_masks[idx]), dim=-1)
            for idx, ph in enumerate(self.policy_head)
        ]
        vals = self.vf_head(logits)[..., 0]

        return action_probs, vals

    def push_processed_obs_to_batch(self, batch_index, batch_size, processed_obs):
        name = f"{_PROCESSED_OBSERVATIONS}_batch_{self.policy}"
        if not self.env.cuda_data_manager.is_data_on_device_via_torch(name):
            processed_obs_batch = np.zeros((batch_size,) + processed_obs.shape)
            processed_obs_feed = DataFeed()
            processed_obs_feed.add_data(name=name, data=processed_obs_batch)
            self.env.cuda_data_manager.push_data_to_device(
                processed_obs_feed, torch_accessible=True
            )
        self.env.cuda_data_manager.data_on_device_via_torch(name=name)[
            batch_index
        ] = processed_obs
