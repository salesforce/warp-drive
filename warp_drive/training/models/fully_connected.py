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
import torch.nn as nn
import torch.nn.functional as F
from gym.spaces import Discrete, MultiDiscrete

from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed

_OBSERVATIONS = Constants.OBSERVATIONS


# Policy networks
# ---------------
class FullyConnected(nn.Module):
    """
    Fully connected network implementation in Pytorch
    """

    name = "torch_fully_connected"

    def __init__(self, env, model_config, policy, policy_tag_to_agent_id_map):
        super().__init__()

        self.env = env
        fc_dims = model_config["fc_dims"]
        assert isinstance(fc_dims, list)
        num_fc_layers = len(fc_dims)
        self.policy = policy
        self.policy_tag_to_agent_id_map = policy_tag_to_agent_id_map
        self.fast_forward_mode = False

        # Flatten obs space
        sample_agent_id = self.policy_tag_to_agent_id_map[self.policy][0]
        obs_space = np.prod(self.env.env.observation_space[sample_agent_id].shape)

        if isinstance(self.env.env.action_space[sample_agent_id], Discrete):
            action_space = [self.env.env.action_space[sample_agent_id].n]
        elif isinstance(self.env.env.action_space[sample_agent_id], MultiDiscrete):
            action_space = self.env.env.action_space[sample_agent_id].nvec
        else:
            raise NotImplementedError

        input_dims = [obs_space] + fc_dims[:-1]
        output_dims = fc_dims

        self.fc = nn.ModuleDict()
        for fc_layer in range(num_fc_layers):
            self.fc[str(fc_layer)] = nn.Sequential(
                nn.Linear(input_dims[fc_layer], output_dims[fc_layer]),
                nn.ReLU(),
            )

        # policy network (list of heads)
        policy_heads = [None for _ in range(len(action_space))]
        for idx, act_space in enumerate(action_space):
            policy_heads[idx] = nn.Linear(fc_dims[-1], act_space)
        self.policy_head = nn.ModuleList(policy_heads)

        # value-function network head
        self.vf_head = nn.Linear(fc_dims[-1], 1)

    def set_fast_forward_mode(self):
        # if there is only one policy with discrete action space,
        # then there is no need to map to agents
        self.fast_forward_mode = True
        print(
            f"the model {self.name} turns on the fast_forward_mode to speed up "
            "the forward calculation (there is only one policy with discrete "
            "action space, therefore in the model forward there is no need to have "
            "an explicit mapping to agents which is slow) "
        )

    def forward(self, batch_index=None, batch_size=None, obs=None):
        if batch_index is not None:
            assert obs is None
            assert batch_index < batch_size
            obs = self.env.cuda_data_manager.data_on_device_via_torch(_OBSERVATIONS)
            # Push obs to obs_batch
            name = f"{_OBSERVATIONS}_batch"
            if not self.env.cuda_data_manager.is_data_on_device_via_torch(name):
                obs_batch = np.zeros((batch_size,) + obs.shape)
                obs_feed = DataFeed()
                obs_feed.add_data(name=name, data=obs_batch)
                self.env.cuda_data_manager.push_data_to_device(
                    obs_feed, torch_accessible=True
                )
            if not self.fast_forward_mode:
                agent_ids_for_policy = self.policy_tag_to_agent_id_map[self.policy]
                self.env.cuda_data_manager.data_on_device_via_torch(name=name)[
                    batch_index, :, agent_ids_for_policy
                ] = obs[:, agent_ids_for_policy]
                ip = obs[:, agent_ids_for_policy]
            else:
                self.env.cuda_data_manager.data_on_device_via_torch(name=name)[
                    batch_index
                ] = obs
                ip = obs
        else:
            assert obs is not None
            ip = obs

        # Feed through the FC layers
        for layer in range(len(self.fc)):
            op = self.fc[str(layer)](ip)
            ip = op

        # Compute the action probabilities and the value function estimate
        probs = [F.softmax(ph(op), dim=-1) for ph in self.policy_head]
        vals = self.vf_head(op)

        return probs, vals[..., 0]
