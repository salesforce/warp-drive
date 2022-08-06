import time
from typing import Optional
import importlib

import numpy as np
import numba.cuda as numba_driver
import torch

from warp_drive.managers.function_manager import CUDAFunctionManager, CUDASampler, CUDAEnvironmentReset
from warp_drive.managers.numba_managers.numba_data_manager import NumbaDataManager
from warp_drive.utils.common import get_project_root
from warp_drive.utils.numba_utils.misc import (
    check_env_header,
    update_env_header,
    update_env_runner,
)
from warp_drive.utils.env_registrar import EnvironmentRegistrar


class NumbaFunctionManager(CUDAFunctionManager):

    def __init__(
            self,
            num_agents: int = 1,
            num_envs: int = 1,
            blocks_per_env: int = 1,
            process_id: int = 0,
    ):
        super().__init__(num_agents=num_agents,
                         num_envs=num_envs,
                         blocks_per_env=blocks_per_env,
                         process_id=process_id)
        self._NUMBA_module = None
        # functions from the numba_managers module
        self._numba_functions = {}
        self._numba_function_names = []

    def import_numba_from_source_code(self, numba_path: str, default_functions_included: bool = True):
        assert (
                self._NUMBA_module is None
        ), "NUMBA module has already been loaded, not allowed to load twice"

        self._NUMBA_module = importlib.import_module(numba_path)
        logging.info("Successfully import the source code")
        if default_functions_included:
            self.initialize_default_functions()

    def dynamic_import_numba(
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
        numba_path = "warp_drive.numba_includes.env_runner"
        header_path = f"{get_project_root()}/warp_drive/numba_includes"
        if template_path is None:
            template_path = f"{get_project_root()}/warp_drive/numba_includes"
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
            header_file="env_config.py",
            path=header_path,
            num_envs=self._num_envs,
            num_agents=self._num_agents,
            blocks_per_env=self._blocks_per_env,
        )
        logging.debug(
            f"header file {header_path}/env_config.py "
            f"has number_agents: {self._num_agents}, "
            f"num_agents per block: {self.block[0]}, "
            f"num_envs: {self._num_envs}, num of blocks: {self.grid[0]} "
            f"and blocks_per_env: {self._blocks_per_env}"
            f"that are consistent with the block and the grid"
        )

        self.import_numba_from_source_code(
            numba_path=numba_path, default_functions_included=default_functions_included
        )

    def initialize_default_functions(self):
        """
        Default function list defined in the src/core. They can be initialized if
        the CUDA compilation includes src/core
        """
        default_func_names = [
            "NumbaRandomService",
            "reset_when_done_2d",
            "reset_when_done_3d",
            "undo_done_flag_and_reset_timestep",
        ]
        self.initialize_functions(default_func_names)
        self._default_functions_initialized = True
        logging.info(
            "Successfully initialize the default NUMBA functions "
            "managed by the NumbaFunctionManager"
        )

    def initialize_functions(self, func_names: Optional[list] = None):
        """
        :param func_names: list of kernel function names in the cuda mdoule
        """
        assert self._NUMBA_module is not None, (
            "NUMBA module has not yet been imported, "
            "call import_numba_from_source_code(file) "
        )
        for fname in func_names:
            assert fname not in self._numba_function_names
            logging.info(
                f"starting to load the numba_managers kernel function: {fname} "
                f"from the NUMBA module "
            )
            self._numba_functions[fname] = getattr(self._NUMBA_module, fname)
            self._numba_function_names.append(fname)
            logging.info(
                f"finished loading the numba_managers kernel function: {fname} "
                f"from the NUMBA module, "
            )

    def _get_function(self, fname):
        """
        :param fname: function name
        return: the CUDA function callable by Python
        """
        assert fname in self._numba_function_names, f"{fname} is not defined"

        return self._numba_functions[fname]

    @property
    def numba_function_names(self):
        return self._numba_function_names


class NumbaSampler(CUDASampler):
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

    def __init__(self, function_manager: NumbaFunctionManager):
        """
        :param function_manager: CUDAFunctionManager object
        """
        super().__init__(function_manager)

        self.numba_random_class = self._function_manager.get_function("NumbaRandomService")()

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
        init = self.numba_random_class.init_random
        init[self._grid, self._block](seed)
        self._random_initialized = True

    def sample(
            self,
            data_manager: NumbaDataManager,
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
            numba_driver.as_cuda_array(distribution),
            data_manager.device_data(action_name),
            data_manager.device_data(f"{action_name}_cum_distr"),
            np.int32(n_agents),
            np.int32(n_actions),
            block=((n_agents - 1) // self._blocks_per_env + 1, 1, 1),
            grid=self._grid,
        )


class NumbaEnvironmentReset(CUDAEnvironmentReset):
    """
    CUDA Environment Reset: Manages the env reset when the game is terminated
    inside GPU. With this, the GPU can automatically reset and
    restart example_envs by itself.

    prerequisite: CUDAFunctionManager is initialized, and the default function list
    has been successfully launched

    Example:
        Please refer to tutorials
    """

    def __init__(self, function_manager: NumbaFunctionManager):
        """
        :param function_manager: CUDAFunctionManager object
        """
        super().__init__(function_manager)

        self.reset_func_2d = self._function_manager.get_function(
            "reset_when_done_2d"
        )
        self.reset_func_3d = self._function_manager.get_function(
            "reset_when_done_3d"
        )
        self.undo = self._function_manager.get_function(
            "undo_done_flag_and_reset_timestep"
        )

    def custom_reset(self,
                     args: Optional[list] = None,
                     block=None,
                     grid=None):

        assert self._cuda_custom_reset is not None and self._cuda_reset_feed is not None, \
            "Custom Reset function is not defined, call register_custom_reset_function() first"
        assert args is None or isinstance(args, list)
        if block is None:
            block = self._block
        if grid is None:
            grid = self._grid
        if args is None or len(args) == 0:
            self._cuda_custom_reset[grid, block]()
        else:
            self._cuda_custom_reset[grid, block](*self._cuda_reset_feed(args))

    def reset_when_done_deterministic(
            self,
            data_manager: NumbaDataManager,
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
            if "float" not in dtype and "int" not in dtype:
                raise Exception(f"unknown dtype: {dtype}"
                                )
            if is_3d:
                reset_func = self.reset_func_3d
                reset_func[self._grid, (int((agent_dim - 1) // self._blocks_per_env + 1), 1, 1)](
                    data_manager.device_data(name),
                    data_manager.device_data(f"{name}_at_reset"),
                    data_manager.device_data("_done_"),
                    agent_dim,
                    feature_dim,
                    force_reset
                )
            else:
                reset_func = self.reset_func_2d
                reset_func[self._grid, (int((feature_dim - 1) // self._blocks_per_env + 1), 1, 1)](
                    data_manager.device_data(name),
                    data_manager.device_data(f"{name}_at_reset"),
                    data_manager.device_data("_done_"),
                    feature_dim,
                    force_reset
                )

        if undo_done_after_reset:
            self._undo_done_flag_and_reset_timestep(data_manager, force_reset)

    def _undo_done_flag_and_reset_timestep(
            self, data_manager: NumbaDataManager, force_reset
    ):
        self.undo[self._grid, (1,1,1)](
            data_manager.device_data("_done_"),
            data_manager.device_data("_timestep_"),
            force_reset
        )
