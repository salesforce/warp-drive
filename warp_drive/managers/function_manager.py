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
    CUDA Function Manager: manages the CUDA module
    and the kernel functions defined therein

    Example:

        cuda_function_manager = CUDAFunctionManager(num_agents=10, num_envs=5)

        # if load from a source code directly
        cuda_function_manager.load_cuda_from_source_code(code)

        # if load from a pre-compiled bin
        cuda_function_manager.load_cuda_from_binary_file(fname)

        # if compile a template source code (so num_agents and num_envs
        can be populated at compile time)
        cuda_function_manager.compile_and_load_cuda(template_header_file)

        cuda_function_manager.initialize_functions(["step", "test"])

        cuda_step_func = cuda_function_manager.get_function("step")

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
    CUDA Sampler: controls probability sampling inside GPU.
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
    CUDA Environment Reset: Manages the env reset when the game is terminated
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

    def register_custom_reset_function(self, data_manager: CUDADataManager, reset_function_name=None):
        if reset_function_name is None or reset_function_name not in self._function_manager._cuda_function_names:
            return
        self._cuda_custom_reset = self._function_manager.get_function(reset_function_name)
        self._cuda_reset_feed = CUDAFunctionFeed(data_manager)

    def custom_reset(self,
                     args: Optional[list] = None,
                     block=None,
                     grid=None):
        raise NotImplementedError

    def reset_when_done(
        self,
        data_manager: CUDADataManager,
        mode: str = "if_done",
        undo_done_after_reset: bool = True,
        use_random_reset: bool = False,
    ):
        if not use_random_reset:
            self.reset_when_done_deterministic(
                data_manager, mode, undo_done_after_reset
            )
        else:
            # TODO: To be implemented
            # self.reset_when_done_random(data_manager, mode, undo_done_after_reset)
            raise NotImplementedError

    def reset_when_done_deterministic(
        self,
        data_manager: CUDADataManager,
        mode: str = "if_done",
        undo_done_after_reset: bool = True,
    ):
        raise NotImplementedError

    def _undo_done_flag_and_reset_timestep(
        self, data_manager: CUDADataManager, force_reset
    ):
        raise NotImplementedError
