# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The env wrapper class
"""

import numpy as np

from warp_drive.managers.data_manager import CUDADataManager
from warp_drive.managers.function_manager import (
    CUDAEnvironmentReset,
    CUDAFunctionManager,
)
from warp_drive.utils.common import get_project_root
from warp_drive.utils.recursive_obs_dict_to_spaces_dict import (
    recursive_obs_dict_to_spaces_dict,
)

_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"


class EnvWrapper:
    """
    The environment wrapper class.
    This wrapper determines whether the environment reset and steps happen on the
    CPU or the GPU, and proceeds accordingly.
    If the environment runs on the CPU, the reset() and step() calls also occur on
    the CPU
    If the environment runs on the GPU, only the first reset() happens on the CPU,
    all the relevant data is copied over the GPU after, and the subsequent steps
    all happen on the GPU
    """

    def __init__(self, env_obj=None, num_envs=1, use_cuda=False, testing_mode=False):
        """
        'env_obj': an environment instance
        'use_cuda': if True, step through the environment on the GPU, else on the CPU
        'num_envs': the number of parallel environments to instantiate. Note: this is
        only relevant when use_cuda is True
        'testing_mode': a flag used to determine whether to simply load the .cubin (when
        testing) or compile the .cu source code to create a .cubin and use that.
        """
        # Need to pass in an environment instance
        assert env_obj is not None
        self.env = env_obj

        self.n_agents = self.env.num_agents
        self.episode_length = self.env.episode_length

        assert self.env.name
        self.name = self.env.name

        # Determine observation and action spaces
        obs = self.env.reset()
        self.env.observation_space = recursive_obs_dict_to_spaces_dict(obs)

        # CUDA-specific initializations
        # -----------------------------
        # Flag to determine whether to use CUDA or not
        self.use_cuda = use_cuda
        self.env.use_cuda = use_cuda

        # Flag to determine where the reset happens (host or device)
        # First reset is always on the host (CPU), and subsequent resets are on
        # the device (GPU)
        self.reset_on_host = True

        if self.use_cuda:
            print("USING CUDA...")

            # Number of environments to run in parallel
            assert num_envs >= 1
            self.n_envs = num_envs

            print("Initializing the CUDA data manager...")
            self.cuda_data_manager = CUDADataManager(
                num_agents=self.n_agents,
                episode_length=self.episode_length,
                num_envs=self.n_envs,
            )

            print("Initializing the CUDA function manager...")
            self.cuda_function_manager = CUDAFunctionManager(
                num_agents=int(self.cuda_data_manager.meta_info("n_agents")),
                num_envs=int(self.cuda_data_manager.meta_info("n_envs")),
            )

            print(f"Using cubin_filepath: {_CUBIN_FILEPATH}")
            if testing_mode:
                self.cuda_function_manager.load_cuda_from_binary_file(
                    f"{_CUBIN_FILEPATH}/test_build.cubin"
                )
            else:
                self.cuda_function_manager.compile_and_load_cuda(
                    env_name=self.name,
                    template_header_file="template_env_config.h",
                    template_runner_file="template_env_runner.cu",
                )

            # Register the CUDA step() function for the env
            # Note: generate_observation() and compute_reward()
            # should be part of the step function itself
            step_function = f"Cuda{self.name}Step"
            self.cuda_function_manager.initialize_functions([step_function])

            # Add wrapper attributes for use within env
            self.env.cuda_data_manager = self.cuda_data_manager
            self.env.cuda_function_manager = self.cuda_function_manager
            self.env.cuda_step = self.cuda_function_manager._get_function(step_function)

            # Register the env resetter
            self.env_resetter = CUDAEnvironmentReset(
                function_manager=self.cuda_function_manager
            )

    def reset_all_envs(self):
        """
        Reset the state of the environment to initialize a new episode.
        if self.reset_on_host is True:
            calls the CPU env to prepare and return the initial state
        if self.use_cuda is True:
            if self.reset_on_host is True:
                expands initial state to parallel example_envs and push to GPU once
                sets self.reset_on_host = False
            else:
                calls device hard reset managed by the CUDAResetter

        """
        self.env.timestep = 0

        if self.reset_on_host:
            # Produce observation
            obs = self.env.reset()
        else:
            assert self.use_cuda

        if self.use_cuda:  # GPU version
            if self.reset_on_host:

                # Helper function to repeat data across the env dimension
                def repeat_across_env_dimension(array, num_envs):
                    return np.stack([array for _ in range(num_envs)], axis=0)

                # Copy host data and tensors to device
                # Note: this happens only once after the first reset on the host

                # Add env dimension to data if "save_copy_and_apply_at_reset" is True
                data_dictionary = self.env.get_data_dictionary()
                tensor_dictionary = self.env.get_tensor_dictionary()
                for key in data_dictionary:
                    if data_dictionary[key]["attributes"][
                        "save_copy_and_apply_at_reset"
                    ]:
                        data_dictionary[key]["data"] = repeat_across_env_dimension(
                            data_dictionary[key]["data"], self.n_envs
                        )

                for key in tensor_dictionary:
                    if tensor_dictionary[key]["attributes"][
                        "save_copy_and_apply_at_reset"
                    ]:
                        tensor_dictionary[key]["data"] = repeat_across_env_dimension(
                            tensor_dictionary[key]["data"], self.n_envs
                        )

                self.cuda_data_manager.push_data_to_device(data_dictionary)

                self.cuda_data_manager.push_data_to_device(
                    tensor_dictionary, torch_accessible=True
                )

                # All subsequent resets happen on the GPU
                self.reset_on_host = False

                # Return the obs
                return obs
            # Returns an empty dictionary for all subsequent resets on the GPU
            # as arrays are modified in place
            self.env_resetter.reset_when_done(
                self.cuda_data_manager, mode="force_reset"
            )
            return {}
        return obs  # CPU version

    def reset_only_done_envs(self):
        """
        This function only works for GPU example_envs.
        It will check all the running example_envs,
        and only resets those example_envs that are observing done flag is True
        """
        assert self.use_cuda and not self.reset_on_host, (
            "reset_only_done_envs() only works "
            "for self.use_cuda = True and self.reset_on_host = False"
        )

        self.env_resetter.reset_when_done(self.cuda_data_manager, mode="if_done")
        return {}

    def step(self, actions=None):
        """
        Step through the environment
        """
        if self.use_cuda:
            self.env.step()
            obs = {}
            rew = {}
            done = {
                "__all__": self.cuda_data_manager.data_on_device_via_torch("_done_") > 0
            }
            info = {}
        else:
            assert actions is not None
            obs, rew, done, info = self.env.step(actions)

        return obs, rew, done, info
