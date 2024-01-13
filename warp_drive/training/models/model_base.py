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
from warp_drive.training.utils.data_loader import get_flattened_obs_size

_OBSERVATIONS = Constants.OBSERVATIONS
_PROCESSED_OBSERVATIONS = Constants.PROCESSED_OBSERVATIONS
_ACTION_MASK = Constants.ACTION_MASK

_LARGE_NEG_NUM = -1e20


class ModelBaseFullyConnected(nn.Module):
    """
    Fully connected network implementation in Pytorch
    """

    name = "model_base_fully_connected"

    def __init__(
        self,
        env,
        model_config,
        policy,
        policy_tag_to_agent_id_map,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
        include_policy_head=True,
        include_value_head=True,
    ):
        super().__init__()

        self.env = env
        self.fc_dims = model_config["fc_dims"]
        assert isinstance(self.fc_dims, list)
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
        self.flattened_obs_size = get_flattened_obs_size(self.observation_space)
        self.is_deterministic = False
        if isinstance(self.env.env.action_space[sample_agent_id], Discrete):
            action_space = [self.env.env.action_space[sample_agent_id].n]
        elif isinstance(self.env.env.action_space[sample_agent_id], MultiDiscrete):
            action_space = self.env.env.action_space[sample_agent_id].nvec
        elif isinstance(self.env.env.action_space[sample_agent_id], Box):
            # deterministic action space
            action_space = [1] * self.env.env.action_space[sample_agent_id].shape[0]
            self.is_deterministic = True
        else:
            raise NotImplementedError

        self.flattened_action_size = len(action_space)

        self.fc = nn.ModuleDict() # this is defined in the child class

        if include_policy_head:
            # policy network (list of heads)
            if self.is_deterministic:
                self.output_dims = [len(action_space)]
                self.policy_head = nn.Linear(self.fc_dims, len(action_space))
            else:
                policy_heads = [None for _ in range(len(action_space))]
                self.output_dims = []  # Network output dimension(s)
                for idx, act_space in enumerate(action_space):
                    self.output_dims += [act_space]
                    policy_heads[idx] = nn.Linear(self.fc_dims[-1], act_space)
                self.policy_head = nn.ModuleList(policy_heads)
        if include_value_head:
            # value-function network head
            self.vf_head = nn.Linear(self.fc_dims[-1], 1)

        # used for action masking
        self.action_mask = None

        # max batch size allowed
        name = f"{_PROCESSED_OBSERVATIONS}_batch_{self.policy}"
        self.batch_size = self.env.cuda_data_manager.get_shape(name=name)[0]

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

        assert flattened_obs.shape[-1] == self.flattened_obs_size, \
            f"The flattened observation size of {flattened_obs.shape[-1]} is different " \
            f"from the designated size of {self.flattened_obs_size} "

        return flattened_obs

    def process_one_step_obs(self):
        obs = self.get_flattened_obs()
        if not self.create_separate_placeholders_for_each_policy:
            agent_ids_for_policy = self.policy_tag_to_agent_id_map[self.policy]
            obs = obs[:, agent_ids_for_policy]
        return obs

    def forward(self, obs=None, action=None):
        raise NotImplementedError

    def push_processed_obs_to_batch(self, batch_index, processed_obs):
        if batch_index >= 0:
            assert batch_index < self.batch_size, f"batch_index: {batch_index}, self.batch_size: {self.batch_size}"
            name = f"{_PROCESSED_OBSERVATIONS}_batch_{self.policy}"
            self.env.cuda_data_manager.data_on_device_via_torch(name=name)[
                batch_index
            ] = processed_obs


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
