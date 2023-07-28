# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import logging
import os
import subprocess
import time
from typing import Optional

import numpy as np
import torch

# TODO: remove this once numba_managers works on independent sampler and reset
from warp_drive.managers.data_manager import CUDADataManager
from warp_drive.utils.data_feed import DataFeed


class CUDAFunctionManager:
    """
    Base CUDA Function Manager: manages the CUDA module
    and the kernel functions defined therein

    """

    def __init__(
        self,
        num_agents: int = 1,
        num_envs: int = 1,
        blocks_per_env: int = 1,
        process_id: int = 0,
    ):
        """
        :param num_agents: total number of agents for each env,
            it defines the default block size if blocks_per_env = 1
            and links to the CUDA constant `wkNumberAgents`
        :param num_envs: number of example_envs in parallel,
            it defines the default grid size if blocks_per_env = 1
            and links to the CUDA constant `wkNumberEnvs`
        :param blocks_per_env: number of blocks to cover one environment
            it links to the CUDA constant `wkBlocksPerEnv`
        :param process_id: device ID

        """
        if num_agents % blocks_per_env != 0:
            logging.warning(
                """
                `num_agents` cannot be divisible by `blocks_per_env`.
                Therefore, the running threads for the last block could
                possibly EXCEED the boundaries of the output arrays and
                incurs index our-of-range bugs.
                Consider to have a proper thread index boundary check,
                for example if you have already checked
                `if (kThisAgentId < NumAgents)`, please ignore this warning.
                """
            )

        self._num_agents = int(num_agents)
        self._num_envs = int(num_envs)
        self._blocks_per_env = int(blocks_per_env)
        self._process_id = process_id
        # number of threads in each block, a ceiling operation is performed
        self._block = (int((self._num_agents - 1) // self._blocks_per_env + 1), 1, 1)
        # number of blocks
        self._grid = (int(self._num_envs * self._blocks_per_env), 1)
        self._default_functions_initialized = False

    def initialize_default_functions(self):
        raise NotImplementedError

    def initialize_functions(self, func_names: Optional[list] = None):
        raise NotImplementedError

    def _get_function(self, fname):
        raise NotImplementedError

    @property
    def get_function(self):
        return self._get_function

    @property
    def block(self):
        return self._block

    @property
    def grid(self):
        return self._grid

    @property
    def blocks_per_env(self):
        return self._blocks_per_env


class CUDAFunctionFeed:
    """
    CUDAFunctionFeed as the intermediate layer to feed data arguments
    into the CUDA function. Please make sure that the order of data
    aligns with the CUDA function signature.

    """

    def __init__(self, data_manager: CUDADataManager):
        self.data_manager = data_manager
        self._function_feeds = None

    def __call__(self, arguments: list) -> list:
        """
        :param arguments: a list of arguments containing names of data
            that are stored inside CUDADataManager

        returns: the list of data pointers as arguments to feed into the CUDA function
        """
        # if the function has not yet defined function feed
        if self._function_feeds is None:
            data_pointers = []
            for arg in arguments:
                if isinstance(arg, str):
                    data_pointers.append(self.data_manager.device_data(arg))
                elif isinstance(arg, tuple):
                    key = arg[0]
                    source = arg[1].lower()
                    if source in ("d", "device"):
                        data_pointers.append(self.data_manager.device_data(key))
                    elif source in ("m", "meta"):
                        data_pointers.append(self.data_manager.meta_info(key))
                    elif source in ("s", "shared"):
                        data_pointers.append(self.data_manager.shared_constant(key))
                else:
                    raise Exception(f"Unknown definition of CUDA function feed: {arg}")
            self._function_feeds = data_pointers

        return self._function_feeds


class CUDASampler:
    """
    Base CUDA Sampler: controls probability sampling inside GPU.
    A fast and lightweight implementation compared to the
    functionality provided by torch.Categorical.sample()
    It accepts the Pytorch tensor as distribution and gives out the sampled action index

    prerequisite: CUDAFunctionManager is initialized,
    and the default function list has been successfully launched

    Example:
        Please refer to tutorials
    """

    def __init__(self, function_manager: CUDAFunctionManager):
        """
        :param function_manager: CUDAFunctionManager object
        """
        self._function_manager = function_manager
        assert self._function_manager._default_functions_initialized, (
            "Default CUDA functions are required to initialized "
            "before SampleController can work, "
            "You may call function_manager.initialize_default_functions() to proceed"
        )
        self._block = function_manager.block
        self._grid = function_manager.grid
        self._blocks_per_env = function_manager.blocks_per_env
        self._num_envs = function_manager._num_envs
        self._random_initialized = False

    def init_random(self, seed: Optional[int] = None):
        raise NotImplementedError

    def register_actions(
        self, data_manager: CUDADataManager, action_name: str, num_actions: int
    ):
        """
        Register an action
        :param data_manager: CUDADataManager object
        :param action_name: the name of action array that will
        record the sampled actions
        :param num_actions: the number of actions for this action_name
        (the last dimension of the action distribution)
        """
        n_agents = data_manager.get_shape(action_name)[1]
        host_array = np.zeros(
            shape=(self._grid[0], n_agents, num_actions), dtype=np.float32
        )
        data_feed = DataFeed()
        data_feed.add_data(name=f"{action_name}_cum_distr", data=host_array)
        data_manager.push_data_to_device(data_feed)

    def sample(
        self,
        data_manager: CUDADataManager,
        distribution: torch.Tensor,
        action_name: str,
    ):
        raise NotImplementedError


class CUDAEnvironmentReset:
    """
    Base CUDA Environment Reset: Manages the env reset when the game is terminated
    inside GPU. With this, the GPU can automatically reset and
    restart example_envs by itself.

    prerequisite: CUDAFunctionManager is initialized, and the default function list
    has been successfully launched

    Example:
        Please refer to tutorials
    """

    def __init__(self, function_manager: CUDAFunctionManager):
        """
        :param function_manager: CUDAFunctionManager object
        """
        self._function_manager = function_manager
        assert self._function_manager._default_functions_initialized, (
            "Default CUDA functions are required to initialized "
            "before EnvironmentReset can work, "
            "You may call function_manager.initialize_default_functions() to proceed"
        )
        self._block = function_manager.block
        self._grid = function_manager.grid
        self._blocks_per_env = function_manager.blocks_per_env
        self._cuda_custom_reset = None
        self._cuda_reset_feed = None
        self._random_initialized = False

    def register_custom_reset_function(
        self, data_manager: CUDADataManager, reset_function_name=None
    ):
        raise NotImplementedError

    def custom_reset(self, args: Optional[list] = None, block=None, grid=None):
        raise NotImplementedError

    def init_reset_pool(
        self,
        data_manager: CUDADataManager,
        seed: Optional[int] = None,
    ):
        raise NotImplementedError

    def reset_when_done(
        self,
        data_manager: CUDADataManager,
        mode: str = "if_done",
        undo_done_after_reset: bool = True,
    ):
        if mode == "if_done":
            force_reset = np.int32(0)
        elif mode == "force_reset":
            force_reset = np.int32(1)
        else:
            raise Exception(
                f"unknown reset mode: {mode}, only accept 'if_done' and 'force_reset' "
            )
        self.reset_when_done_deterministic(data_manager, force_reset)
        self.reset_when_done_from_pool(data_manager, force_reset)
        if undo_done_after_reset:
            self._undo_done_flag_and_reset_timestep(data_manager, force_reset)

    def reset_when_done_deterministic(
        self,
        data_manager: CUDADataManager,
        force_reset: int,
    ):
        raise NotImplementedError

    def reset_when_done_from_pool(
        self,
        data_manager: CUDADataManager,
        force_reset: int,
    ):
        raise NotImplementedError

    def _undo_done_flag_and_reset_timestep(
        self, data_manager: CUDADataManager, force_reset
    ):
        raise NotImplementedError


class CUDALogController:
    """
    Base CUDA Log Controller: manages the CUDA logger inside GPU for all the data having
    the flag log_data_across_episode = True.
    The log function will only work for one particular env, even there are multiple
    example_envs running together.

    prerequisite: CUDAFunctionManager is initialized, and the default function list
    has been successfully launched

    Example:
        Please refer to tutorials

    """

    def __init__(self, function_manager: CUDAFunctionManager):
        """
        :param function_manager: CUDAFunctionManager object
        """
        self._function_manager = function_manager
        assert self._function_manager._default_functions_initialized, (
            "Default CUDA functions are required to initialized "
            "before LogController can work, "
            "You may call function_manager.initialize_default_functions() to proceed"
        )
        self._block = function_manager.block
        self._grid = function_manager.grid
        self._blocks_per_env = function_manager.blocks_per_env
        self.last_valid_step = -1
        self._env_id = None

    def update_log(self, data_manager: CUDADataManager, step: int):
        """
        Update the log for all the data having the flag log_data_across_episode = True

        :param data_manager: CUDADataManager object
        :param step: the logging step
        """
        assert (
            step > self.last_valid_step
        ), "update_log is trying to update the existing timestep"
        self._log_one_step(data_manager, step, self._env_id)
        self._update_log_mask(data_manager, step)

    def reset_log(self, data_manager: CUDADataManager, env_id: int = 0):
        """
        Reset the dense log mask back to [1, 0, 0, 0 ....]

        :param data_manager: CUDADataManager object
        :param env_id: the env with env_id will reset log and later update_log()
        will be executed for this env.
        """
        self._env_id = env_id
        self.last_valid_step = -1
        logging.info(f"reset log for env {self._env_id}")
        self._reset_log_mask(data_manager)
        self.update_log(data_manager, step=0)

    def fetch_log(
        self,
        data_manager: CUDADataManager,
        names: Optional[str] = None,
        last_step: Optional[int] = None,
        check_last_valid_step: bool = True,
    ):
        """
        Fetch the complete log back to the host.

        :param data_manager: CUDADataManager object
        :param names: names of the data
        :param last_step: optional, if provided, return data till min(last_step, )
        :param check_last_valid_step: if True, check if host and device are consistent
        with the last_valid_step

        returns: the log at the host
        """
        if check_last_valid_step is True:
            self._cuda_check_last_valid_step(data_manager)

        if last_step is not None and last_step <= self.last_valid_step:
            last_valid_step = last_step
        else:
            last_valid_step = self.last_valid_step

        data = {}
        if names is None:
            names = data_manager.log_data_list

        for name in names:
            name = f"{name}_for_log"
            d = data_manager.pull_data_from_device(name)
            assert len(d) == int(data_manager.meta_info("episode_length")) + 1
            data[name] = d[: last_valid_step + 1]
        return data

    def _log_one_step(self, data_manager: CUDADataManager, step: int, env_id: int = 0):
        raise NotImplementedError

    def _update_log_mask(self, data_manager: CUDADataManager, step: int):
        """
        Mark the success of the current step and assign 1 for the dense_log_mask,
        update self.last_valid_step
        """
        raise NotImplementedError

    def _reset_log_mask(self, data_manager: CUDADataManager):
        raise NotImplementedError

    def _cuda_check_last_valid_step(self, data_manager: CUDADataManager):
        """
        Check if self.last_valid_step maintained by step() is consistent
        with dense_log_mask
        """
        log_mask = data_manager.pull_data_from_device("_log_mask_")
        pos_1s = np.argwhere(log_mask == 1).reshape(-1)
        pos_0s = np.argwhere(log_mask == 0).reshape(-1)
        if len(pos_1s) > 0 and len(pos_0s) > 0 and pos_0s[0] < pos_1s[-1]:
            raise Exception("there is invalid log data in the middle")
        if len(pos_1s) > 0:
            last_valid_step = pos_1s[-1]
        else:
            last_valid_step = -1

        assert last_valid_step == self.last_valid_step, (
            f"inconsistency of last_valid_step derived from "
            f"dense_log_mask = {last_valid_step} "
            f"and the step() function = {self.last_valid_step}"
        )
