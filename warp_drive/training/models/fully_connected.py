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
from torch import nn
from warp_drive.training.models.model_base import ModelBaseFullyConnected, apply_logit_mask


# Policy + Value networks
# ---------------
class FullyConnected(ModelBaseFullyConnected):
    """
    Fully connected network implementation in Pytorch
    """

    name = "torch_fully_connected"

    def __init__(
        self,
        env,
        model_config,
        policy,
        policy_tag_to_agent_id_map,
        create_separate_placeholders_for_each_policy=False,
        obs_dim_corresponding_to_num_agents="first",
    ):
        super().__init__(env,
                         model_config,
                         policy,
                         policy_tag_to_agent_id_map,
                         create_separate_placeholders_for_each_policy,
                         obs_dim_corresponding_to_num_agents,)
        num_fc_layers = len(self.fc_dims)
        input_dims = [self.flattened_obs_size] + self.fc_dims[:-1]
        output_dims = self.fc_dims
        for fc_layer in range(num_fc_layers):
            self.fc[str(fc_layer)] = nn.Sequential(
                nn.Linear(input_dims[fc_layer], output_dims[fc_layer]),
                nn.ReLU(),
            )

    def forward(self, obs=None, action=None):
        """
        Forward pass through the model.
        Returns action probabilities and value functions.
        """
        ip = obs
        # Feed through the FC layers
        for layer in range(len(self.fc)):
            op = self.fc[str(layer)](ip)
            ip = op
        logits = op

        # Compute the action probabilities and the value function estimate
        # Apply action mask to the logits as well.
        if self.is_deterministic:
            combined_action_probs = func.tanh(apply_logit_mask(self.policy_head(logits), self.action_mask))
            combined_action_probs = self.action_scale * combined_action_probs + self.action_bias
            if self.output_dims[0] > 1:
                # we split the actions to their corresponding heads
                # we make sure after the split, we rearrange the memory so each chunk is still C-continguous
                # otherwise the sampler may have index issue
                action_probs = list(torch.split(combined_action_probs, 1, dim=-1))
                action_probs = [ap.contiguous() for ap in action_probs]
            else:
                action_probs = [combined_action_probs]
        else:
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

