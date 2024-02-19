# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
#

import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np

from warp_drive.training.utils.param_scheduler import ParamScheduler

_EPSILON = 1e-10  # small number to prevent indeterminate division


class DDPG:
    """
    The Advantage Actor-Critic Class
    https://arxiv.org/abs/1602.01783
    """

    def __init__(
        self,
        discount_factor_gamma=1.0,
        normalize_advantage=False,
        normalize_return=False,
        n_step=1,
    ):
        assert 0 <= discount_factor_gamma <= 1
        assert n_step >= 1
        self.discount_factor_gamma = discount_factor_gamma
        self.normalize_advantage = normalize_advantage
        self.normalize_return = normalize_return
        self.n_step = n_step

    def compute_loss_and_metrics(
        self,
        timestep=None,
        actions_batch=None,
        rewards_batch=None,
        done_flags_batch=None,
        value_functions_batch=None,
        next_value_functions_batch=None,
        j_functions_batch=None,
        perform_logging=False,
    ):
        assert actions_batch is not None
        assert timestep is not None
        assert rewards_batch is not None
        assert done_flags_batch is not None
        assert value_functions_batch is not None
        assert next_value_functions_batch is not None
        assert j_functions_batch is not None

        # we only calculate up to batch - n_step + 1 point,
        # after that it is not enough points to calculate n_step
        valid_batch_range = rewards_batch.shape[0] - self.n_step + 1
        # Detach value_functions_batch from the computation graph
        # for return and advantage computations.
        next_value_functions_batch_detached = next_value_functions_batch.detach()

        # Value objective.
        returns_batch = torch.zeros_like(rewards_batch[:valid_batch_range])

        for i in range(valid_batch_range):
            last_step = i + self.n_step - 1
            if last_step < rewards_batch.shape[0] - 1:
                r = rewards_batch[last_step] + \
                    (1 - done_flags_batch[last_step][:, None]) * \
                    self.discount_factor_gamma * next_value_functions_batch_detached[last_step]
            else:
                r = done_flags_batch[last_step][:, None] * rewards_batch[last_step] + \
                    (1 - done_flags_batch[last_step][:, None]) * \
                    self.discount_factor_gamma * next_value_functions_batch_detached[-1]
            for j in range(1, self.n_step):
                r = (1 - done_flags_batch[last_step - j][:, None]) * self.discount_factor_gamma * r + \
                    done_flags_batch[last_step - j][:, None] * torch.zeros_like(rewards_batch[last_step - j])
                r += rewards_batch[last_step - j]
            returns_batch[i] = r
        # Normalize across the agents and env dimensions
        if self.normalize_return:
            normalized_returns_batch = (
                returns_batch - returns_batch.mean(dim=(1, 2), keepdim=True)
            ) / (returns_batch.std(dim=(1, 2), keepdim=True) + torch.tensor(_EPSILON))
        else:
            normalized_returns_batch = returns_batch

        value_functions_batch = value_functions_batch[:valid_batch_range]
        critic_loss = nn.MSELoss()(normalized_returns_batch, value_functions_batch)

        advantages_batch = normalized_returns_batch - value_functions_batch

        # Normalize across the agents and env dimensions
        if self.normalize_advantage:
            normalized_advantages_batch = (
                advantages_batch - advantages_batch.mean(dim=(1, 2), keepdim=True)
            ) / (
                advantages_batch.std(dim=(1, 2), keepdim=True) + torch.tensor(_EPSILON)
            )
        else:
            normalized_advantages_batch = advantages_batch

        # Policy objective
        j_functions_batch = j_functions_batch[:valid_batch_range]
        if self.normalize_return:
            normalized_j_functions_batch = (
                j_functions_batch - j_functions_batch.mean(dim=(1, 2), keepdim=True)
            ) / (
                j_functions_batch.std(dim=(1, 2), keepdim=True) + torch.tensor(_EPSILON))
        else:
            normalized_j_functions_batch = j_functions_batch

        actor_loss = -normalized_j_functions_batch.mean()

        variance_explained = max(
            torch.tensor(-1.0),
            (
                1
                - (
                    normalized_advantages_batch.detach().var()
                    / (normalized_returns_batch.detach().var() + torch.tensor(_EPSILON))
                )
            ),
        )

        if perform_logging:
            metrics = {
                "Total loss": actor_loss.item() + critic_loss.item(),
                "Actor loss": actor_loss.item(),
                "Critic loss": critic_loss.item(),
                "Mean rewards": rewards_batch.mean().item(),
                "Max. rewards": rewards_batch.max().item(),
                "Min. rewards": rewards_batch.min().item(),
                "Mean value function": value_functions_batch.mean().item(),
                "Mean J function": j_functions_batch.mean().item(),
                "Mean advantages": advantages_batch.mean().item(),
                "Mean (norm.) advantages": normalized_advantages_batch.mean().item(),
                "Mean (discounted) returns": returns_batch.mean().item(),
                "Mean normalized returns": normalized_returns_batch.mean().item(),
                "Variance explained by the value function": variance_explained.item(),
            }
            # mean of the standard deviation of sampled actions
            std_over_agent_per_action = (
                actions_batch.float().std(axis=2).mean(axis=(0, 1))
            )
            std_over_time_per_action = (
                actions_batch.float().std(axis=0).mean(axis=(0, 1))
            )
            std_over_env_per_action = (
                actions_batch.float().std(axis=1).mean(axis=(0, 1))
            )
            max_per_action = torch.amax(actions_batch, dim=(0, 1, 2))
            min_per_action = torch.amin(actions_batch, dim=(0, 1, 2))

            for idx, _ in enumerate(std_over_agent_per_action):
                std_action = {
                    f"Std. of action_{idx} over agents": std_over_agent_per_action[
                        idx
                    ].item(),
                    f"Std. of action_{idx} over envs": std_over_env_per_action[
                        idx
                    ].item(),
                    f"Std. of action_{idx} over time": std_over_time_per_action[
                        idx
                    ].item(),
                    f"Max of action_{idx}": max_per_action[
                        idx
                    ].item(),
                    f"Min of action_{idx}": min_per_action[
                        idx
                    ].item(),
                }
                metrics.update(std_action)
        else:
            metrics = {}
        return actor_loss, critic_loss, metrics
