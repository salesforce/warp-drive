# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

"""
The env wrapper class
"""

import logging

import numpy as np

from warp_drive.managers.function_manager import CUDAFunctionFeed
from warp_drive.utils.argument_fix import Argfix
from warp_drive.utils.common import get_project_root
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext
from warp_drive.utils.recursive_obs_dict_to_spaces_dict import (
    recursive_obs_dict_to_spaces_dict,
)

_CUBIN_FILEPATH = f"{get_project_root()}/warp_drive/cuda_bin"
_NUMBA_FILEPATH = "warp_drive.numba_includes"


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

    Note: Versions <= 1.7.0 has `use_cuda = True or False`. For users who are using the
    old API for their application but have the new library installed, we add a runtime
    arg fixer that if old API arg is seen by the new library, it will raise a warning
    and convert to the new syntax. It will not do anything otherwise.
    """

    @Argfix(arg_mapping={"use_cuda": "env_backend"})
    def __init__(
        self,
        env_obj=None,
        env_name=None,
        env_config=None,
        num_envs=1,
        blocks_per_env=None,
        env_backend="cpu",
        testing_mode=False,
        testing_bin_filename=None,
        env_registrar=None,
        event_messenger=None,
        process_id=0,
    ):
        """
        :param env_obj: an environment object
        :param env_name: an environment name that is registered on the
            WarpDrive environment registrar
        :param env_config: environment configuration to instantiate
            an environment from the registrar
        :param num_envs: the number of parallel environments to instantiate. Note: this is
            only relevant when env_backend is pycuda or numba
        :param blocks_per_env: number of blocks to cover one environment
            default is None, the utility function will estimate it
            otherwise it will be reinforced
        :param env_backend: environment backend, choose between pycuda, numba, or cpu.
            Before version 2.0, the old argument is 'use_cuda' = True or False
        :param use_cuda: deprecated since version 1.8
        :param testing_mode: a flag used to determine whether to simply load the .cubin (when
            testing) or compile the .cu source code to create a .cubin and use that.
        :param testing_bin_filename: load the specified .cubin or .fatbin directly,
            only required when testing_mode is True.
        :param env_registrar: EnvironmentRegistrar object
            it provides the customized env info (like src path) for the build
        :param event_messenger: multiprocessing Event to sync up the build
            when using multiple processes
        :param process_id: id of the process running WarpDrive
        """

        # backward compatibility with old argument use_cuda = True or False
        if isinstance(env_backend, bool):
            env_backend = "pycuda" if env_backend is True else "cpu"

        # Need to pass in an environment instance
        if env_obj is not None:
            self.env = env_obj
        else:
            assert (
                env_name is not None
                and env_config is not None
                and env_registrar is not None
            )
            self.env = env_registrar.get(env_name, env_backend)(**env_config)

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
        # Flag to determine which backend to use
        if env_backend not in ("pycuda", "numba", "cpu"):
            logging.warn("Environment backend not recognized, defaulting to cpu")
            env_backend = "cpu"
        self.env_backend = env_backend
        if hasattr(self.env, "env_backend"):
            self.env.env_backend = env_backend

        # Flag to determine where the reset happens (host or device)
        # First reset is always on the host (CPU), and subsequent resets are on
        # the device (GPU)
        self.reset_on_host = True

        # Steps specific to GPU runs
        # --------------------------
        if not self.env_backend == "cpu":
            logging.info("USING CUDA...")

            assert isinstance(self.env, CUDAEnvironmentContext), (
                f"{self.env_backend} backend requires the environment "
                f"an instance of CUDAEnvironmentContext"
            )

            # Number of environments to run in parallel
            assert num_envs >= 1
            self.n_envs = num_envs
            if blocks_per_env is not None:
                self.blocks_per_env = blocks_per_env
            else:
                if self.env_backend == "pycuda":
                    from warp_drive.utils.architecture_validate import (
                        calculate_blocks_per_env,
                    )

                    self.blocks_per_env = calculate_blocks_per_env(self.n_agents)
                else:
                    self.blocks_per_env = 1
            logging.info(f"We use blocks_per_env = {self.blocks_per_env} ")

            if self.env_backend == "pycuda":
                from warp_drive.managers.pycuda_managers.pycuda_data_manager import (
                    PyCUDADataManager,
                )
                from warp_drive.managers.pycuda_managers.pycuda_function_manager import (
                    PyCUDAEnvironmentReset,
                    PyCUDAFunctionManager,
                )

                backend_data_manager = PyCUDADataManager
                backend_function_manager = PyCUDAFunctionManager
                backend_env_resetter = PyCUDAEnvironmentReset
            elif self.env_backend == "numba":
                from warp_drive.managers.numba_managers.numba_data_manager import (
                    NumbaDataManager,
                )
                from warp_drive.managers.numba_managers.numba_function_manager import (
                    NumbaEnvironmentReset,
                    NumbaFunctionManager,
                )

                backend_data_manager = NumbaDataManager
                backend_function_manager = NumbaFunctionManager
                backend_env_resetter = NumbaEnvironmentReset

            logging.info("Initializing the CUDA data manager...")
            self.cuda_data_manager = backend_data_manager(
                num_agents=self.n_agents,
                episode_length=self.episode_length,
                num_envs=self.n_envs,
                blocks_per_env=self.blocks_per_env,
            )

            logging.info("Initializing the CUDA function manager...")
            self.cuda_function_manager = backend_function_manager(
                num_agents=int(self.cuda_data_manager.meta_info("n_agents")),
                num_envs=int(self.cuda_data_manager.meta_info("n_envs")),
                blocks_per_env=int(self.cuda_data_manager.meta_info("blocks_per_env")),
                process_id=process_id,
            )
            if self.env_backend == "pycuda":
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
            elif self.env_backend == "numba":
                if testing_mode:
                    logging.info(f"Using numba_filepath: {_NUMBA_FILEPATH}")
                    assert self.n_agents == 5
                    assert self.n_envs == 2
                    assert self.blocks_per_env == 1
                    self.cuda_function_manager.dynamic_import_numba(
                        env_name=self.name,
                        template_header_file="template_env_config.txt",
                        template_runner_file="template_env_runner.txt",
                    )
                else:
                    self.cuda_function_manager.dynamic_import_numba(
                        env_name=self.name,
                        template_header_file="template_env_config.txt",
                        template_runner_file="template_env_runner.txt",
                        customized_env_registrar=env_registrar,
                        event_messenger=event_messenger,
                    )

            self.cuda_function_feed = CUDAFunctionFeed(self.cuda_data_manager)

            # Register the CUDA step() function for the env
            # Note: generate_observation() and compute_reward()
            # should be part of the step function itself
            step_function = (
                f"Cuda{self.name}Step"
                if self.env_backend == "pycuda"
                else f"Numba{self.name}Step"
            )
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
            self.env_resetter = backend_env_resetter(
                function_manager=self.cuda_function_manager
            )
            # custom reset function, if not found, will ignore
            reset_function = f"Cuda{self.name}Reset"
            self.env_resetter.register_custom_reset_function(
                self.cuda_data_manager, reset_function_name=reset_function
            )

    def reset_all_envs(self):
        """
        Reset the state of the environment to initialize a new episode.
        if self.reset_on_host is True:
            calls the CPU env to prepare and return the initial state
        if self.env_backend is pycuda or numba:
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
            assert not self.env_backend == "cpu"

        if not self.env_backend == "cpu":  # GPU version
            if self.reset_on_host:

                # Helper function to repeat data across the env dimension
                def repeat_across_env_dimension(array, num_envs):
                    return np.stack([array for _ in range(num_envs)], axis=0)

                # Copy host data and tensors to device
                # Note: this happens only once after the first reset on the host

                data_dictionary = self.env.get_data_dictionary()
                tensor_dictionary = self.env.get_tensor_dictionary()
                reset_pool_dictionary = self.env.get_reset_pool_dictionary()
                # Add env dimension to data if "save_copy_and_apply_at_reset" is True
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
                # Add env dimension to data if "is_reset_pool" exists for this data
                # if so, also check this data has "save_copy_and_apply_at_reset" = False
                for key in reset_pool_dictionary:
                    if "is_reset_pool" in reset_pool_dictionary[key]["attributes"] and \
                            reset_pool_dictionary[key]["attributes"]["is_reset_pool"]:
                        # find the corresponding target data
                        reset_target = reset_pool_dictionary[key]["attributes"]["reset_target"]
                        if reset_target in data_dictionary:
                            assert not data_dictionary[reset_target]["attributes"]["save_copy_and_apply_at_reset"]
                            data_dictionary[reset_target]["data"] = repeat_across_env_dimension(
                                data_dictionary[reset_target]["data"], self.n_envs
                            )
                        elif reset_target in tensor_dictionary:
                            assert not tensor_dictionary[reset_target]["attributes"]["save_copy_and_apply_at_reset"]
                            tensor_dictionary[reset_target]["data"] = repeat_across_env_dimension(
                                tensor_dictionary[reset_target]["data"], self.n_envs
                            )
                        else:
                            raise Exception(f"Fail to locate the target data {reset_target} for the reset pool "
                                            f"in neither data_dictionary nor tensor_dictionary")

                self.cuda_data_manager.push_data_to_device(data_dictionary)

                self.cuda_data_manager.push_data_to_device(
                    tensor_dictionary, torch_accessible=True
                )

                self.cuda_data_manager.push_data_to_device(reset_pool_dictionary)

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

    def init_reset_pool(self, seed=None):
        self.env_resetter.init_reset_pool(self.cuda_data_manager, seed)

    def reset_only_done_envs(self):
        """
        This function only works for GPU example_envs.
        It will check all the running example_envs,
        and only resets those example_envs that are observing done flag is True
        """
        assert (not self.env_backend == "cpu") and not self.reset_on_host, (
            "reset_only_done_envs() only works "
            "for pycuda or numba backends and self.reset_on_host = False"
        )

        self.env_resetter.reset_when_done(self.cuda_data_manager, mode="if_done")
        return {}

    def custom_reset_all_envs(self, args=None, block=None, grid=None):
        self.env_resetter.custom_reset(args=args, block=block, grid=grid)
        return {}

    def step_all_envs(self, actions=None):
        """
        Step through all the environments
        """
        if not self.env_backend == "cpu":
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
