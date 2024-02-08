# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause
#

import torch
from torch import nn
from torch.distributions import Categorical

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
    ):
        assert 0 <= discount_factor_gamma <= 1
        self.discount_factor_gamma = discount_factor_gamma
        self.normalize_advantage = normalize_advantage
        self.normalize_return = normalize_return

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

        # Detach value_functions_batch from the computation graph
        # for return and advantage computations.
        next_value_functions_batch_detached = next_value_functions_batch.detach()

        # Value objective.
        returns_batch = torch.zeros_like(rewards_batch)

        returns_batch[-1] = (
            done_flags_batch[-1][:, None] * rewards_batch[-1]
            + (1 - done_flags_batch[-1][:, None]) * next_value_functions_batch_detached[-1]
        )
        returns_batch[:-1] = rewards_batch[:-1] + \
            self.discount_factor_gamma * (1 - done_flags_batch[:-1][:, :, None]) * next_value_functions_batch_detached

        # Normalize across the agents and env dimensions
        if self.normalize_return:
            normalized_returns_batch = (
                returns_batch - returns_batch.mean(dim=(1, 2), keepdim=True)
            ) / (returns_batch.std(dim=(1, 2), keepdim=True) + torch.tensor(_EPSILON))
        else:
            normalized_returns_batch = returns_batch

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
