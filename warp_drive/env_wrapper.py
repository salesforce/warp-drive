# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The env wrapper class
"""

import logging

import numpy as np

from warp_drive.managers.data_manager import CUDADataManager
from warp_drive.managers.function_manager import (
    CUDAEnvironmentReset,
    CUDAFunctionFeed,
    CUDAFunctionManager,
)
from warp_drive.utils.architecture_validate import calculate_blocks_per_env
from warp_drive.utils.common import get_project_root
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext
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
    the CPU.
    If the environment runs on the GPU, only the first reset() happens on the CPU,
    all the relevant data is copied over the GPU after, and the subsequent steps
    all happen on the GPU.
    """

    def __init__(
        self,
        env_obj=None,
        env_name=None,
        env_config=None,
        num_envs=1,
        blocks_per_env=None,
        use_cuda=False,
        testing_mode=False,
        testing_bin_filename=None,
        env_registrar=None,
        event_messenger=None,
        process_id=0,
    ):
        """
        'env_obj': an environment object
        'env_name': an environment name that is registered on the
            WarpDrive environment registrar
        'env_config': environment configuration to instantiate
            an environment from the registrar
        'use_cuda': if True, step through the environment on the GPU, else on the CPU
        'num_envs': the number of parallel environments to instantiate. Note: this is
            only relevant when use_cuda is True
        'blocks_per_env': number of blocks to cover one environment
            default is None, the utility function will estimate it
            otherwise it will be reinforced
        'testing_mode': a flag used to determine whether to simply load the .cubin (when
            testing) or compile the .cu source code to create a .cubin and use that.
        'testing_bin_filename': load the specified .cubin or .fatbin directly,
            only required when testing_mode is True.
        'env_registrar': EnvironmentRegistrar object
            it provides the customized env info (like src path) for the build
        'event_messenger': multiprocessing Event to sync up the build
            when using multiple processes
        'process_id': id of the process running WarpDrive
        """
        # Need to pass in an environment instance
        if env_obj is not None:
            self.env = env_obj
        else:
            assert (
                env_name is not None
                and env_config is not None
                and env_registrar is not None
            )
            self.env = env_registrar.get(env_name, use_cuda)(**env_config)

        self.n_agents = self.env.num_agents
        self.episode_length = self.env.episode_length

        assert self.env.name
        self.name = self.env.name

        # Add observation space to the env
        obs = self.obs_at_reset()
        self.env.observation_space = recursive_obs_dict_to_spaces_dict(obs)
        # Ensure the observation and action spaces share the same keys
        assert set(self.env.observation_space.keys()) == set(
            self.env.action_space.keys()
        )

        # CUDA-specific initializations
        # -----------------------------
        # Flag to determine whether to use CUDA or not
        self.use_cuda = use_cuda
        if hasattr(self.env, "use_cuda"):
            self.env.use_cuda = use_cuda

        # Flag to determine where the reset happens (host or device)
        # First reset is always on the host (CPU), and subsequent resets are on
        # the device (GPU)
        self.reset_on_host = True

        # Steps specific to GPU runs
        # --------------------------
        if self.use_cuda:
            logging.info("USING CUDA...")

            assert isinstance(
                self.env, CUDAEnvironmentContext
            ), "use_cuda requires the environment an instance of CUDAEnvironmentContext"

            # Number of environments to run in parallel
            assert num_envs >= 1
            self.n_envs = num_envs
            if blocks_per_env is not None:
                self.blocks_per_env = blocks_per_env
            else:
                self.blocks_per_env = calculate_blocks_per_env(self.n_agents)
            logging.info(f"We use blocks_per_env = {self.blocks_per_env} ")

            logging.info("Initializing the CUDA data manager...")
            self.cuda_data_manager = CUDADataManager(
                num_agents=self.n_agents,
                episode_length=self.episode_length,
                num_envs=self.n_envs,
                blocks_per_env=self.blocks_per_env,
            )

            logging.info("Initializing the CUDA function manager...")
            self.cuda_function_manager = CUDAFunctionManager(
                num_agents=int(self.cuda_data_manager.meta_info("n_agents")),
                num_envs=int(self.cuda_data_manager.meta_info("n_envs")),
                blocks_per_env=int(self.cuda_data_manager.meta_info("blocks_per_env")),
                process_id=process_id,
            )

            if testing_mode:
                logging.info(f"Using cubin_filepath: {_CUBIN_FILEPATH}")
                if testing_bin_filename is None:
                    testing_bin_filename = "test_build.fatbin"
                assert (
                    ".cubin" in testing_bin_filename
                    or ".fatbin" in testing_bin_filename
                )
                self.cuda_function_manager.load_cuda_from_binary_file(
                    f"{_CUBIN_FILEPATH}/{testing_bin_filename}"
                )
            else:
                self.cuda_function_manager.compile_and_load_cuda(
                    env_name=self.name,
                    template_header_file="template_env_config.h",
                    template_runner_file="template_env_runner.cu",
                    customized_env_registrar=env_registrar,
                    event_messenger=event_messenger,
                )
            self.cuda_function_feed = CUDAFunctionFeed(self.cuda_data_manager)

            # Register the CUDA step() function for the env
            # Note: generate_observation() and compute_reward()
            # should be part of the step function itself
            step_function = f"Cuda{self.name}Step"
            context_ready = self.env.initialize_step_function_context(
                cuda_data_manager=self.cuda_data_manager,
                cuda_function_manager=self.cuda_function_manager,
                cuda_step_function_feed=self.cuda_function_feed,
                step_function_name=step_function,
            )
            assert (
                context_ready
            ), "The environment class failed to initialize the CUDA step function"
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
            obs = self.obs_at_reset()
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

    def step_all_envs(self, actions=None):
        """
        Step through all the environments
        """
        if self.use_cuda:
            self.env.step()
            result = None  # Do not return anything
        else:
            assert actions is not None, "Please provide actions to step with."
            result = self.env.step(actions)
        return result

    def obs_at_reset(self):
        """
        Calls the (Python) env to reset and return the initial state
        """
        return self.env.reset()

    def reset(self):
        """
        Alias for reset_all_envs() when CPU is used (conforms to gym-style)
        """
        return self.reset_all_envs()

    def step(self, actions=None):
        """
        Alias for step_all_envs() when CPU is used (conforms to gym-style)
        """
        return self.step_all_envs(actions)
