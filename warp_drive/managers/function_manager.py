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
import pycuda.driver as cuda_driver
import torch
from pycuda.compiler import SourceModule
from pycuda.driver import Context

from warp_drive.managers.data_manager import CUDADataManager, CudaTensorHolder
from warp_drive.utils.architecture_validate import validate_device_setup
from warp_drive.utils.common import (
    check_env_header,
    get_project_root,
    update_env_header,
    update_env_runner,
)
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.env_registrar import EnvironmentRegistrar


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
        self._CUDA_module = None

        # functions from the cuda module
        self._cuda_functions = {}
        self._cuda_function_names = []
        self._num_agents = int(num_agents)
        self._num_envs = int(num_envs)
        self._blocks_per_env = int(blocks_per_env)
        self._process_id = process_id
        # number of threads in each block, a ceiling operation is performed
        self._block = (int((self._num_agents - 1) // self._blocks_per_env + 1), 1, 1)
        # number of blocks
        self._grid = (int(self._num_envs * self._blocks_per_env), 1)
        self._default_functions_initialized = False

        cc = Context.get_device().compute_capability()  # compute capability
        self.arch = f"sm_{cc[0]}{cc[1]}"
        valid = validate_device_setup(
            arch=self.arch,
            num_blocks=self._grid[0],
            threads_per_block=self._block[0],
            blocks_per_env=self._blocks_per_env,
        )
        if not valid:
            raise Exception("The simulation setup fails to pass the validation")

    def load_cuda_from_source_code(
        self, code: str, default_functions_included: bool = True
    ):
        """
        Load cuda module from the source code
        NOTE: the source code is in string text format,
        not the directory of the source code.
        :param code: source code in the string text format
        :param default_functions_included: load default function lists
        """
        assert (
            self._CUDA_module is None
        ), "CUDA module has already been loaded, not allowed to load twice"

        self._CUDA_module = SourceModule(code, no_extern_c=True)

        logging.info("Successfully build and load the source code")
        if default_functions_included:
            self.initialize_default_functions()

    def load_cuda_from_binary_file(
        self, cubin: str, default_functions_included: bool = True
    ):
        """
        Load cuda module from the pre-compiled cubin file

        :param cubin: the binary (.cubin) directory
        :param default_functions_included: load default function lists
        """
        assert (
            self._CUDA_module is None
        ), "CUDA module has already been loaded, not allowed to load twice"

        self._CUDA_module = cuda_driver.module_from_file(cubin)
        logging.info(f"Successfully load the cubin_file from {cubin}")
        if default_functions_included:
            self.initialize_default_functions()

    def compile_and_load_cuda(
        self,
        env_name: str,
        template_header_file: str,
        template_runner_file: str,
        template_path: Optional[str] = None,
        default_functions_included: bool = True,
        customized_env_registrar: Optional[EnvironmentRegistrar] = None,
        event_messenger=None,
    ):
        """
        Compile a template source code, so self.num_agents and self.num_envs
        will replace the template code at compile time.
        Note: self.num_agents: total number of agents for each env,
        it defines the default block size
        self.num_envs: number of example_envs in parallel,
            it defines the default grid size

        :param env_name: name of the environment for the build
        :param template_header_file: template header,
            e.g., "template_env_config.h"
        :param template_runner_file: template runner,
            e.g., "template_env_runner.cu"
        :param template_path: template path, by default,
        it is f"{ROOT_PATH}/warp_drive/cuda_includes/"
        :param default_functions_included: load default function lists
        :param customized_env_registrar: CustomizedEnvironmentRegistrar object
            it provides the customized env info (e.g., source code path)for the build
        :param event_messenger: multiprocessing Event to sync up the build
        when using multiple processes
        """

        # cubin_file is the targeted compiled exe
        bin_path = f"{get_project_root()}/warp_drive/cuda_bin"
        cubin_file = f"{bin_path}/env_runner.fatbin"

        # Only process 0 is taking care of the compilation
        if self._process_id > 0:
            assert event_messenger is not None, (
                "Event messenger is required to sync up "
                "the compilation status among processes."
            )
            event_messenger.wait(timeout=12)

            if not event_messenger.is_set():
                raise Exception(
                    f"Process {self._process_id} fails to get "
                    f"the successful compilation message ... "
                )

        else:
            # 'bin_path' is the designated cuda exe binary path that warp_drive
            # is built into; 'header_path' is the designated cuda main source code path
            # that warp_drive is trying to build.
            # DO NOT CHANGE THEM!
            header_path = f"{get_project_root()}/warp_drive/cuda_includes"
            if template_path is None:
                template_path = f"{get_project_root()}/warp_drive/cuda_includes"
            update_env_header(
                template_header_file=template_header_file,
                path=template_path,
                num_agents=self._num_agents,
                num_envs=self._num_envs,
                blocks_per_env=self._blocks_per_env,
            )
            update_env_runner(
                template_runner_file=template_runner_file,
                path=template_path,
                env_name=env_name,
                customized_env_registrar=customized_env_registrar,
            )
            check_env_header(
                header_file="env_config.h",
                path=header_path,
                num_envs=self._num_envs,
                num_agents=self._num_agents,
                blocks_per_env=self._blocks_per_env,
            )
            logging.debug(
                f"header file {header_path}/env_config.h "
                f"has number_agents: {self._num_agents}, "
                f"num_agents per block: {self.block[0]}, "
                f"num_envs: {self._num_envs}, num of blocks: {self.grid[0]} "
                f"and blocks_per_env: {self._blocks_per_env}"
                f"that are consistent with the block and the grid"
            )

            # main_file is the source code
            main_file = f"{header_path}/env_runner.cu"

            logging.info(f"Compiling {main_file} -> {cubin_file}")

            self._compile(main_file, cubin_file, arch=self.arch)

            if event_messenger is not None:
                event_messenger.set()

        self.load_cuda_from_binary_file(
            cubin=cubin_file, default_functions_included=default_functions_included
        )

    @staticmethod
    def _compile(main_file, cubin_file, arch=None):

        bin_path = f"{get_project_root()}/warp_drive/cuda_bin"
        mkbin = f"mkdir -p {bin_path}"

        with subprocess.Popen(
            mkbin, shell=True, stderr=subprocess.STDOUT
        ) as mkbin_process:
            if mkbin_process.wait() != 0:
                raise Exception("make bin file failed ... ")
        logging.info(f"Successfully mkdir the binary folder {bin_path}")

        if os.path.exists(f"{cubin_file}"):
            os.remove(f"{cubin_file}")

        try:
            if arch is None:
                cc = Context.get_device().compute_capability()  # compute capability
                arch = f"sm_{cc[0]}{cc[1]}"
            cmd = f"nvcc --fatbin -arch={arch} {main_file} -o {cubin_file}"
            with subprocess.Popen(
                cmd, shell=True, stderr=subprocess.STDOUT
            ) as make_process:
                if make_process.wait() != 0:
                    raise Exception(
                        f"build failed when running the following build... : \n"
                        f"{cmd} \n"
                        f"try to build the fatbin hybrid version "
                        f"of virtual PTX + gpu binary ... "
                    )
            logging.info(f"Running cmd: {cmd}")
            logging.info(
                f"Successfully build the cubin_file "
                f"from {main_file} to {cubin_file}"
            )
            return

        except Exception as err:
            logging.error(err)

        arch_codes = [
            "-code=sm_37",
            "-code=sm_50",
            "-code=sm_60",
            "-code=sm_70",
            "-code=sm_80",
        ]
        compiler = "nvcc --fatbin -arch=compute_37 -code=compute_37"
        in_out_fname = f"{main_file} -o {cubin_file}"
        # for example, cmd = f"nvcc --fatbin -arch=compute_30 -code=sm_30 -code=sm_50 "
        #                    f"-code=sm_60 -code=sm_70 {main_file} -o {cubin_file}"
        build_success = False
        for i in range(len(arch_codes)):
            try:
                cmd = " ".join(
                    [compiler] + arch_codes[: len(arch_codes) - i] + [in_out_fname]
                )
                with subprocess.Popen(
                    cmd, shell=True, stderr=subprocess.STDOUT
                ) as make_process:
                    if make_process.wait() != 0:
                        raise Exception(
                            f"build failed when running the following build... : \n"
                            f"{cmd} \n"
                            f"try to build the lower gpu-code version ... "
                        )
                logging.info(f"Running cmd: {cmd}")
                logging.info(
                    f"Successfully build the cubin_file "
                    f"from {main_file} to {cubin_file}"
                )
                build_success = True
                break
            except Exception as err:
                logging.error(err)

        if not build_success:
            raise Exception("build failed ... ")

    def initialize_default_functions(self):
        """
        Default function list defined in the src/core. They can be initialized if
        the CUDA compilation includes src/core
        """
        default_func_names = [
            "reset_log_mask",
            "update_log_mask",
            "log_one_step_in_float",
            "log_one_step_in_int",
            "reset_in_float_when_done_2d",
            "reset_in_int_when_done_2d",
            "reset_in_float_when_done_3d",
            "reset_in_int_when_done_3d",
            "undo_done_flag_and_reset_timestep",
            "init_random",
            "free_random",
            "sample_actions",
        ]
        self.initialize_functions(default_func_names)
        self._default_functions_initialized = True
        logging.info(
            "Successfully initialize the default CUDA functions "
            "managed by the CUDAFunctionManager"
        )

    def initialize_functions(self, func_names: Optional[list] = None):
        """
        :param func_names: list of kernel function names in the cuda mdoule
        """
        assert self._CUDA_module is not None, (
            "CUDA module has not yet been loaded, "
            "call load_cuda_from_source_code(code), or "
            "load_cuda_from_binary_file(file) first "
        )
        for fname in func_names:
            assert fname not in self._cuda_functions
            assert fname not in self._cuda_function_names
            logging.info(
                f"starting to load the cuda kernel function: {fname} "
                f"from the CUDA module "
            )
            self._cuda_functions[fname] = self._CUDA_module.get_function(fname)
            self._cuda_function_names.append(fname)
            logging.info(
                f"finished loading the cuda kernel function: {fname} "
                f"from the CUDA module, "
            )

    def initialize_shared_constants(
        self, data_manager: CUDADataManager, constant_names: list
    ):
        """
        Initialize the shared constants in the runtime.
        :param data_manager: CUDADataManager object
        :param constant_names: names of constants managed by CUDADataManager
        """
        for cname in constant_names:
            constant_on_device, _ = self._CUDA_module.get_global(cname)
            cuda_driver.memcpy_htod(
                constant_on_device, data_manager.shared_constant(cname)
            )
            logging.info(
                f"Successfully initialize the CUDA shared constant {cname} "
                f"managed by the CUDAFunctionManager"
            )

    def _get_function(self, fname):
        """
        :param fname: function name
        return: the CUDA function callable by Python
        """
        assert fname in self._cuda_function_names, f"{fname} is not defined"

        return self._cuda_functions[fname]

    @property
    def compile(self):
        return self._compile

    @property
    def get_function(self):
        return self._get_function

    @property
    def cuda_function_names(self):
        return self._cuda_function_names

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


class CUDALogController:
    """
    CUDA Log Controller: manages the CUDA logger inside GPU for all the data having
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
        step = np.int32(step)
        assert env_id < data_manager.meta_info("n_envs")
        env_id = np.int32(env_id)

        log_func_in_float = self._function_manager.get_function("log_one_step_in_float")
        log_func_in_int = self._function_manager.get_function("log_one_step_in_int")

        for name in data_manager.log_data_list:
            f_shape = data_manager.get_shape(name)
            assert f_shape[0] == data_manager.meta_info(
                "n_envs"
            ), "log function assumes the 0th dimension is n_envs"
            assert f_shape[1] == data_manager.meta_info(
                "n_agents"
            ), "log function assumes the 1st dimension is n_agents"
            if len(f_shape) >= 3:
                feature_dim = np.int32(np.prod(f_shape[2:]))
            else:
                feature_dim = np.int32(1)
            dtype = data_manager.get_dtype(name)
            if "float" in dtype:
                log_func = log_func_in_float
            elif "int" in dtype:
                log_func = log_func_in_int
            else:
                raise Exception(f"unknown dtype: {dtype}")
            log_func(
                data_manager.device_data(f"{name}_for_log"),
                data_manager.device_data(name),
                feature_dim,
                step,
                data_manager.meta_info("episode_length"),
                env_id,
                block=self._block,
                grid=(self._blocks_per_env, 1),
            )

    def _update_log_mask(self, data_manager: CUDADataManager, step: int):
        """
        Mark the success of the current step and assign 1 for the dense_log_mask,
        update self.last_valid_step
        """
        step = np.int32(step)
        update_mask = self._function_manager.get_function("update_log_mask")
        update_mask(
            data_manager.device_data("_log_mask_"),
            step,
            data_manager.meta_info("episode_length"),
            block=self._block,
            grid=(self._blocks_per_env, 1),
        )
        self.last_valid_step = step

    def _reset_log_mask(self, data_manager: CUDADataManager):
        reset = self._function_manager.get_function("reset_log_mask")
        reset(
            data_manager.device_data("_log_mask_"),
            data_manager.meta_info("episode_length"),
            block=self._block,
            grid=(self._blocks_per_env, 1),
        )

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

        self.sample_actions = self._function_manager.get_function("sample_actions")

    def __del__(self):
        free = self._function_manager.get_function("free_random")
        free(block=self._block, grid=self._grid)
        self._random_initialized = False
        logging.info(
            "CUDASampler has explicitly released the random states memory in CUDA"
        )

    def init_random(self, seed: Optional[int] = None):
        """
        Init random function for all the threads
        :param seed: random seed selected for the initialization
        """
        if seed is None:
            seed = time.time()
            logging.info(
                f"random seed is not provided, by default, "
                f"using the current timestamp {seed} as seed"
            )
        seed = np.int32(seed)
        init = self._function_manager.get_function("init_random")
        init(seed, block=self._block, grid=self._grid)
        self._random_initialized = True

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
        """
        Sample based on the distribution

        :param data_manager: CUDADataManager object
        :param distribution: Torch distribution tensor in the shape of
        (num_env, num_agents, num_actions)
        :param action_name: the name of action array that will
        record the sampled actions
        """
        assert self._random_initialized, (
            "sample() requires the random seed initialized first, "
            "please call init_random()"
        )
        assert torch.is_tensor(distribution)
        assert distribution.shape[0] == self._num_envs
        n_agents = int(distribution.shape[1])
        assert data_manager.get_shape(action_name)[1] == n_agents
        n_actions = distribution.shape[2]
        assert data_manager.get_shape(f"{action_name}_cum_distr")[2] == n_actions

        # distribution is a runtime output from pytorch at device,
        # it should not be managed by data manager because
        # it is a temporary output and never sit at the host
        self.sample_actions(
            CudaTensorHolder(distribution),
            data_manager.device_data(action_name),
            data_manager.device_data(f"{action_name}_cum_distr"),
            np.int32(n_agents),
            np.int32(n_actions),
            block=((n_agents - 1) // self._blocks_per_env + 1, 1, 1),
            grid=self._grid,
        )

    @staticmethod
    def assign(data_manager: CUDADataManager, actions: np.ndarray, action_name: str):
        """
        Assign action to the action array directly. T
        his may be used for env testing or debugging purpose.
        :param data_manager: CUDADataManager object
        :param actions: actions array provided by the user
        :param action_name: the name of action array that will
        record the sampled actions
        """
        assert data_manager.is_data_on_device_via_torch(action_name)
        assert actions.shape == data_manager.get_shape(action_name)
        assert actions.dtype.name == data_manager.get_dtype(action_name)

        data_manager.data_on_device_via_torch(action_name)[:] = torch.from_numpy(
            actions
        ).cuda()


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

        self.reset_func_in_float_2d = self._function_manager.get_function(
            "reset_in_float_when_done_2d"
        )
        self.reset_func_in_int_2d = self._function_manager.get_function(
            "reset_in_int_when_done_2d"
        )
        self.reset_func_in_float_3d = self._function_manager.get_function(
            "reset_in_float_when_done_3d"
        )
        self.reset_func_in_int_3d = self._function_manager.get_function(
            "reset_in_int_when_done_3d"
        )
        self.undo = self._function_manager.get_function(
            "undo_done_flag_and_reset_timestep"
        )

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
        """
        Monitor the done flag for each env. If any env is done, it will reset this
        particular env without interrupting other example_envs.
        The reset includes copy the starting values of this env back,
        and turn off the done flag. Therefore, this env can safely get restarted.

        :param data_manager: CUDADataManager object
        :param mode: "if_done": reset an env if done flag is observed for that env,
                     "force_reset": reset all env in a hard way
        :param undo_done_after_reset: If True, turn off the done flag
        and reset timestep after all data have been reset
        (the flag should be True for most cases)
        """
        if mode == "if_done":
            force_reset = np.int32(0)
        elif mode == "force_reset":
            force_reset = np.int32(1)
        else:
            raise Exception(
                f"unknown reset mode: {mode}, only accept 'if_done' and 'force_reset' "
            )

        for name in data_manager.reset_data_list:
            f_shape = data_manager.get_shape(name)
            assert f_shape[0] == data_manager.meta_info(
                "n_envs"
            ), "reset function assumes the 0th dimension is n_envs"
            if len(f_shape) >= 3:
                agent_dim = np.int32(f_shape[1])
                feature_dim = np.int32(np.prod(f_shape[2:]))
                is_3d = True
            elif len(f_shape) == 2:
                feature_dim = np.int32(f_shape[1])
                is_3d = False
            else:  # len(f_shape) == 1:
                feature_dim = np.int32(1)
                is_3d = False
            dtype = data_manager.get_dtype(name)
            if is_3d:
                if "float" in dtype:
                    reset_func = self.reset_func_in_float_3d
                elif "int" in dtype:
                    reset_func = self.reset_func_in_int_3d
                else:
                    raise Exception(f"unknown dtype: {dtype}")
                reset_func(
                    data_manager.device_data(name),
                    data_manager.device_data(f"{name}_at_reset"),
                    data_manager.device_data("_done_"),
                    agent_dim,
                    feature_dim,
                    force_reset,
                    block=(int((agent_dim - 1) // self._blocks_per_env + 1), 1, 1),
                    grid=self._grid,
                )
            else:
                if "float" in dtype:
                    reset_func = self.reset_func_in_float_2d
                elif "int" in dtype:
                    reset_func = self.reset_func_in_int_2d
                else:
                    raise Exception(f"unknown dtype: {dtype}")
                reset_func(
                    data_manager.device_data(name),
                    data_manager.device_data(f"{name}_at_reset"),
                    data_manager.device_data("_done_"),
                    feature_dim,
                    force_reset,
                    block=(int((feature_dim - 1) // self._blocks_per_env + 1), 1, 1),
                    grid=self._grid,
                )

        if undo_done_after_reset:
            self._undo_done_flag_and_reset_timestep(data_manager, force_reset)

    def _undo_done_flag_and_reset_timestep(
        self, data_manager: CUDADataManager, force_reset
    ):
        self.undo(
            data_manager.device_data("_done_"),
            data_manager.device_data("_timestep_"),
            force_reset,
            block=(1, 1, 1),
            grid=self._grid,
        )
