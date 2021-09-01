# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
#
"""
The Advantage Actor-Critic Class
https://arxiv.org/abs/1602.01783
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical


class A2C:
    def __init__(
        self,
        discount_factor_gamma=1.0,
        normalize_advantage=False,
        normalize_return=False,
        vf_loss_coeff=0.01,
        entropy_coeff=0.01,
    ):
        assert 0 <= discount_factor_gamma <= 1
        self.discount_factor_gamma = discount_factor_gamma
        self.normalize_advantage = normalize_advantage
        self.normalize_return = normalize_return
        self.vf_loss_coeff = vf_loss_coeff
        self.entropy_coeff = entropy_coeff

    def compute_loss_and_metrics(
        self,
        actions_batch=None,
        rewards_batch=None,
        done_flags_batch=None,
        probabilities_batch=None,
        value_function_batch=None,
    ):
        assert actions_batch is not None
        assert rewards_batch is not None
        assert done_flags_batch is not None
        assert probabilities_batch is not None
        assert value_function_batch is not None

        # Policy objective
        advantages_batch = (
            rewards_batch[:-1]
            + self.discount_factor_gamma * value_function_batch[1:]
            - value_function_batch[:-1]
        )

        # Normalize across the agents and env dimensions
        if self.normalize_advantage:
            normalized_advantages_batch = (
                advantages_batch - advantages_batch.mean(dim=(1, 2), keepdim=True)
            ) / (advantages_batch.std(dim=(1, 2), keepdim=True) + 1e-10)
        else:
            normalized_advantages_batch = advantages_batch

        log_prob = 0.0
        mean_entropy = 0.0

        for idx in range(actions_batch.shape[-1]):
            m = Categorical(probabilities_batch[idx])
            mean_entropy += m.entropy().mean()
            log_prob += m.log_prob(actions_batch[..., idx])

        policy_loss = (-log_prob[:-1] * normalized_advantages_batch).mean()

        # Value objective.
        returns_batch = torch.zeros_like(rewards_batch)

        returns_batch[-1] = (
            done_flags_batch[-1][:, None] * rewards_batch[-1]
            + (1 - done_flags_batch[-1][:, None]) * value_function_batch[-1]
        )
        for step in range(-2, -returns_batch.shape[0] - 1, -1):
            future_return = (
                done_flags_batch[step][:, None] * torch.zeros_like(rewards_batch[step])
                + (1 - done_flags_batch[step][:, None])
                * self.discount_factor_gamma
                * returns_batch[step + 1]
            )
            returns_batch[step] = rewards_batch[step] + future_return

        # Normalize across the agents and env dimension
        if self.normalize_return:
            normalized_returns_batch = (
                returns_batch - returns_batch.mean(dim=(1, 2), keepdim=True)
            ) / (returns_batch.std(dim=(1, 2), keepdim=True) + 1e-10)
        else:
            normalized_returns_batch = returns_batch

        vf_loss = nn.MSELoss()(normalized_returns_batch, value_function_batch)

        loss = (
            policy_loss
            + self.vf_loss_coeff * vf_loss
            - self.entropy_coeff * mean_entropy
        )

        variance_explained = max(
            -1.0,
            (
                1
                - (
                    normalized_advantages_batch.detach().var()
                    / normalized_returns_batch.detach().var()
                )
            ).item(),
        )

        metrics = {
            "Total loss": loss.item(),
            "Policy loss": policy_loss.item(),
            "Value function loss": vf_loss.item(),
            "Mean rewards": rewards_batch.mean().item(),
            "Max. rewards": rewards_batch.max().item(),
            "Min. rewards": rewards_batch.min().item(),
            "Mean value function": value_function_batch.mean().item(),
            "Mean advantages": advantages_batch.mean().item(),
            "Mean (normalized) advantages": normalized_advantages_batch.mean().item(),
            "Mean (discounted) returns": returns_batch.mean().item(),
            "Mean normalized returns": normalized_returns_batch.mean().item(),
            "Mean entropy": mean_entropy.item(),
            "Variance explained by the value function": variance_explained,
        }
        return loss, metrics
