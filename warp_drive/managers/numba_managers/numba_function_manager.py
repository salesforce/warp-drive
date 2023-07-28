import importlib
import logging
import time
from typing import Optional

import numba.cuda as numba_driver
import numpy as np
import torch
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32

from warp_drive.managers.function_manager import (
    CUDAEnvironmentReset,
    CUDAFunctionFeed,
    CUDAFunctionManager,
    CUDALogController,
    CUDASampler,
)
from warp_drive.managers.numba_managers.numba_data_manager import NumbaDataManager
from warp_drive.utils.common import get_project_root
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from warp_drive.utils.numba_utils.misc import (
    check_env_header,
    update_env_header,
    update_env_runner,
)


class NumbaFunctionManager(CUDAFunctionManager):
    """"""
    """
    Example:

        numba_function_manager = NumbaFunctionManager(num_agents=10, num_envs=5)

        # if load from a source code directly
        numba_function_manager.import_numba_from_source_code(numba_path)

        # if compile a template source code (so num_agents and num_envs
        can be populated at compile time)
        numba_function_manager.dynamic_import_numba(template_header_file)

        numba_function_manager.initialize_functions(["step", "test"])

        numba_step_func = numba_function_manager.get_function("step")

    """

    def __init__(
        self,
        num_agents: int = 1,
        num_envs: int = 1,
        blocks_per_env: int = 1,
        process_id: int = 0,
    ):
        super().__init__(
            num_agents=num_agents,
            num_envs=num_envs,
            blocks_per_env=blocks_per_env,
            process_id=process_id,
        )
        self._NUMBA_module = None
        # functions from the numba_managers module
        self._numba_functions = {}
        self._numba_function_names = []
        print(f"function_manager: Setting Numba to use CUDA device {process_id}")

    def import_numba_from_source_code(
        self,
        numba_path: str,
        default_functions_included: bool = True,
    ):
        assert (
            self._NUMBA_module is None
        ), "NUMBA module has already been loaded, not allowed to load twice"

        self._NUMBA_module = importlib.import_module(numba_path)
        logging.info("Successfully import the source code")
        if default_functions_included:
            self.initialize_default_functions()

    def import_numba_env_config(
        self,
        template_header_file: str,
        template_path: Optional[str] = None,
    ):
        if template_path is None:
            template_path = f"{get_project_root()}/warp_drive/numba_includes"
        update_env_header(
            template_header_file=template_header_file,
            path=template_path,
            num_agents=self._num_agents,
            num_envs=self._num_envs,
            blocks_per_env=self._blocks_per_env,
        )

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
        Dynamic import a template source code, so self.num_agents and self.num_envs
        will replace the template code at JIT compile time.
        Note: self.num_agents: total number of agents for each env,
        it defines the default block size
        self.num_envs: number of example_envs in parallel,
            it defines the default grid size

        :param env_name: name of the environment for the build
        :param template_header_file: template header,
            e.g., "template_env_config.txt"
        :param template_runner_file: template runner,
            e.g., "template_env_runner.txt"
        :param template_path: template path, by default,
        it is f"{ROOT_PATH}.warp_drive.numba_includes/"
        :param default_functions_included: load default function lists
        :param customized_env_registrar: CustomizedEnvironmentRegistrar object
            it provides the customized env info (e.g., source code path)for the build
        :param event_messenger: multiprocessing Event to sync up the build
        when using multiple processes
        """
        numba_path = "warp_drive.numba_includes.env_runner"
        header_path = f"{get_project_root()}/warp_drive/numba_includes"

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

            if event_messenger is not None:
                event_messenger.set()

        self.import_numba_from_source_code(
            numba_path=numba_path, default_functions_included=default_functions_included
        )

    def initialize_default_functions(self):
        """
        Default function list defined in the numba_includes/core. T
        hey can be initialized if the Numba compilation
        includes numba_includes/core
        """
        default_func_names = [
            "reset_log_mask",
            "update_log_mask",
            "log_one_step_2d",
            "log_one_step_3d",
            "init_random",
            "sample_actions",
            "reset_when_done_1d",
            "reset_when_done_2d",
            "reset_when_done_3d",
            "init_random_for_reset",
            "reset_when_done_1d_from_pool",
            "reset_when_done_2d_from_pool",
            "reset_when_done_3d_from_pool",
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
        :param func_names: list of kernel function names
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
        return: the Numba function callable by Python
        """
        assert fname in self._numba_function_names, f"{fname} is not defined"

        return self._numba_functions[fname]

    @property
    def numba_function_names(self):
        return self._numba_function_names


class NumbaSampler(CUDASampler):
    """
    Numba Sampler: controls probability sampling inside GPU.
    A fast and lightweight implementation compared to the
    functionality provided by torch.Categorical.sample()
    It accepts the Pytorch tensor as distribution and gives out the sampled action index

    prerequisite: NumbaFunctionManager is initialized,
    and the default function list has been successfully launched

    Example:
        Please refer to tutorials
    """

    def __init__(self, function_manager: NumbaFunctionManager):
        """
        :param function_manager: CUDAFunctionManager object
        """
        super().__init__(function_manager)

        self.sample_actions = self._function_manager.get_function("sample_actions")

        self.rng_states_dict = {}

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
        xoroshiro128p_dtype = np.dtype(
            [("s0", np.uint64), ("s1", np.uint64)], align=True
        )
        sz = self._function_manager._num_envs * self._function_manager._num_agents
        rng_states = numba_driver.device_array(sz, dtype=xoroshiro128p_dtype)
        init = self._function_manager.get_function("init_random")
        init(rng_states, seed)
        self.rng_states_dict["rng_states"] = rng_states
        self._random_initialized = True

    def sample(
        self,
        data_manager: NumbaDataManager,
        distribution: torch.Tensor,
        action_name: str,
        use_argmax: bool = False,
    ):
        """
        Sample based on the distribution

        :param data_manager: NumbaDataManager object
        :param distribution: Torch distribution tensor in the shape of
        (num_env, num_agents, num_actions)
        :param action_name: the name of action array that will
        record the sampled actions
        :param use_argmax: if True, sample based on the argmax(distribution)
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
        self.sample_actions[
            self._grid, (int((n_agents - 1) // self._blocks_per_env + 1), 1, 1)
        ](
            self.rng_states_dict["rng_states"],
            numba_driver.as_cuda_array(distribution.detach()),
            data_manager.device_data(action_name),
            data_manager.device_data(f"{action_name}_cum_distr"),
            np.int32(n_actions),
            np.int32(use_argmax),
        )


class NumbaEnvironmentReset(CUDAEnvironmentReset):
    """
    Numba Environment Reset: Manages the env reset when the game is terminated
    inside GPU. With this, the GPU can automatically reset and
    restart example_envs by itself.

    prerequisite: NumbaFunctionManager is initialized, and the default function list
    has been successfully launched

    Example:
        Please refer to tutorials
    """

    def __init__(self, function_manager: NumbaFunctionManager):
        """
        :param function_manager: NumbaFunctionManager object
        """
        super().__init__(function_manager)

        self.reset_func_1d = self._function_manager.get_function("reset_when_done_1d")
        self.reset_func_2d = self._function_manager.get_function("reset_when_done_2d")
        self.reset_func_3d = self._function_manager.get_function("reset_when_done_3d")

        self.reset_func_1d_from_pool = self._function_manager.get_function("reset_when_done_1d_from_pool")
        self.reset_func_2d_from_pool = self._function_manager.get_function("reset_when_done_2d_from_pool")
        self.reset_func_3d_from_pool = self._function_manager.get_function("reset_when_done_3d_from_pool")

        self.undo = self._function_manager.get_function(
            "undo_done_flag_and_reset_timestep"
        )
        self.rng_states_dict = {}

    def register_custom_reset_function(
        self, data_manager: NumbaDataManager, reset_function_name=None
    ):
        if (
            reset_function_name is None
            or reset_function_name not in self._function_manager._numba_function_names
        ):
            return
        self._cuda_custom_reset = self._function_manager.get_function(
            reset_function_name
        )
        self._cuda_reset_feed = CUDAFunctionFeed(data_manager)

    def custom_reset(self, args: Optional[list] = None, block=None, grid=None):

        assert (
            self._cuda_custom_reset is not None and self._cuda_reset_feed is not None
        ), (
            "Custom Reset function is not defined, call "
            "register_custom_reset_function() first"
        )
        assert args is None or isinstance(args, list)
        if block is None:
            block = self._block
        if grid is None:
            grid = self._grid
        if args is None or len(args) == 0:
            self._cuda_custom_reset[grid, block]()
        else:
            self._cuda_custom_reset[grid, block](*self._cuda_reset_feed(args))

    def init_reset_pool(
        self,
        data_manager: NumbaDataManager,
        seed: Optional[int] = None,
    ):
        """
        Init random function for the reset pool
        :param data_manager: NumbaDataManager object
        :param seed: random seed selected for the initialization
        """
        if len(data_manager.reset_target_to_pool) == 0:
            return

        self._security_check_reset_pool(data_manager)

        if seed is None:
            seed = time.time()
            logging.info(
                f"random seed is not provided, by default, "
                f"using the current timestamp {seed} as seed"
            )
        seed = np.int32(seed)
        xoroshiro128p_dtype = np.dtype(
            [("s0", np.uint64), ("s1", np.uint64)], align=True
        )
        sz = self._function_manager._num_envs
        rng_states = numba_driver.device_array(sz, dtype=xoroshiro128p_dtype)
        init_random_for_reset = self._function_manager.get_function("init_random_for_reset")
        init_random_for_reset(rng_states, seed)
        self.rng_states_dict["rng_states"] = rng_states
        self._random_initialized = True

    def _security_check_reset_pool(self, data_manager):
        for name, pool_name in data_manager.reset_target_to_pool.items():
            data_shape = data_manager.get_shape(name)
            data_type = data_manager.get_dtype(name)
            pool_shape = data_manager.get_shape(pool_name)
            pool_type = data_manager.get_dtype(pool_name)
            assert data_type == pool_type, \
                f"Inconsistency of dtype is found for data: {name} has type {data_type} " \
                f"and its reset pool: {pool_name} has type {pool_type}"
            assert data_shape[0] == self._function_manager._num_envs
            assert pool_shape[0] > 1
            for i in range(1, len(data_shape)):
                assert data_shape[i] == pool_shape[i], \
                    f"Inconsistency of shape is found for data: {name} and its reset pool: {pool_name}"

    def reset_when_done_deterministic(
        self,
        data_manager: NumbaDataManager,
        force_reset: int,
    ):
        """
        Monitor the done flag for each env. If any env is done, it will reset this
        particular env without interrupting other example_envs.
        The reset includes copy the starting values of this env back,
        and turn off the done flag. Therefore, this env can safely get restarted.

        :param data_manager: NumbaDataManager object
        :param force_reset: 0: reset an env if done flag is observed for that env,
                            1: reset all env in a hard way
        """
        if len(data_manager.reset_data_list) == 0:
            return

        for name in data_manager.reset_data_list:
            f_shape = data_manager.get_shape(name)
            assert f_shape[0] == data_manager.meta_info(
                "n_envs"
            ), "reset function assumes the 0th dimension is n_envs"
            if len(f_shape) >= 3:
                if len(f_shape) > 3:
                    raise Exception(
                        "Numba environment.reset() temporarily "
                        "not supports array dimension > 3"
                    )
                agent_dim = np.int32(f_shape[1])
                feature_dim = np.int32(np.prod(f_shape[2:]))
                data_shape = "is_3d"
            elif len(f_shape) == 2:
                feature_dim = np.int32(f_shape[1])
                data_shape = "is_2d"
            else:  # len(f_shape) == 1:
                feature_dim = np.int32(1)
                data_shape = "is_1d"

            dtype = data_manager.get_dtype(name)
            if "float" not in dtype and "int" not in dtype:
                raise Exception(f"unknown dtype: {dtype}")
            if data_shape == "is_3d":
                reset_func = self.reset_func_3d
                reset_func[
                    self._grid, (int((agent_dim - 1) // self._blocks_per_env + 1), 1, 1)
                ](
                    data_manager.device_data(name),
                    data_manager.device_data(f"{name}_at_reset"),
                    data_manager.device_data("_done_"),
                    agent_dim,
                    feature_dim,
                    force_reset,
                )
            elif data_shape == "is_2d":
                reset_func = self.reset_func_2d
                reset_func[
                    self._grid,
                    (int((feature_dim - 1) // self._blocks_per_env + 1), 1, 1),
                ](
                    data_manager.device_data(name),
                    data_manager.device_data(f"{name}_at_reset"),
                    data_manager.device_data("_done_"),
                    feature_dim,
                    force_reset,
                )
            elif data_shape == "is_1d":
                reset_func = self.reset_func_1d
                reset_func[self._grid, (1, 1, 1)](
                    data_manager.device_data(name),
                    data_manager.device_data(f"{name}_at_reset"),
                    data_manager.device_data("_done_"),
                    force_reset,
                )

    def reset_when_done_from_pool(
            self,
            data_manager: NumbaDataManager,
            force_reset: int,
    ):
        """
        Monitor the done flag for each env. If any env is done, it will reset this
        particular env without interrupting other example_envs.
        The reset includes randomly select starting values from the candidate pool,
        and copy the starting values of this env back,
        and turn off the done flag. Therefore, this env can safely get restarted.

        :param data_manager: NumbaDataManager object
        :param force_reset: 0: reset an env if done flag is observed for that env,
                            1: reset all env in a hard way
        """
        if len(data_manager.reset_target_to_pool) == 0:
            return

        assert self._random_initialized, (
            "reset_when_done_from_pool() requires the random seed initialized first, "
            "please call init_reset_pool()"
        )

        for name, pool_name in data_manager.reset_target_to_pool.items():
            f_shape = data_manager.get_shape(name)
            assert f_shape[0] > 1, "reset function assumes the 0th dimension is n_pool"
            if len(f_shape) >= 3:
                if len(f_shape) > 3:
                    raise Exception(
                        "Numba environment.reset() temporarily "
                        "not supports array dimension > 3"
                    )
                agent_dim = np.int32(f_shape[1])
                feature_dim = np.int32(np.prod(f_shape[2:]))
                data_shape = "is_3d"
            elif len(f_shape) == 2:
                feature_dim = np.int32(f_shape[1])
                data_shape = "is_2d"
            else:  # len(f_shape) == 1:
                feature_dim = np.int32(1)
                data_shape = "is_1d"

            dtype = data_manager.get_dtype(name)
            if "float" not in dtype and "int" not in dtype:
                raise Exception(f"unknown dtype: {dtype}")
            if data_shape == "is_3d":
                reset_func = self.reset_func_3d_from_pool
                reset_func[
                    self._grid, (int((agent_dim - 1) // self._blocks_per_env + 1), 1, 1)
                ](
                    self.rng_states_dict["rng_states"],
                    data_manager.device_data(name),
                    data_manager.device_data(pool_name),
                    data_manager.device_data("_done_"),
                    agent_dim,
                    feature_dim,
                    force_reset,
                )
            elif data_shape == "is_2d":
                reset_func = self.reset_func_2d_from_pool
                reset_func[
                    self._grid,
                    (int((feature_dim - 1) // self._blocks_per_env + 1), 1, 1),
                ](
                    self.rng_states_dict["rng_states"],
                    data_manager.device_data(name),
                    data_manager.device_data(pool_name),
                    data_manager.device_data("_done_"),
                    feature_dim,
                    force_reset,
                )
            elif data_shape == "is_1d":
                reset_func = self.reset_func_1d_from_pool
                reset_func[self._grid, (1, 1, 1)](
                    self.rng_states_dict["rng_states"],
                    data_manager.device_data(name),
                    data_manager.device_data(pool_name),
                    data_manager.device_data("_done_"),
                    force_reset,
                )

    def _undo_done_flag_and_reset_timestep(
        self, data_manager: NumbaDataManager, force_reset
    ):
        self.undo[self._grid, (1, 1, 1)](
            data_manager.device_data("_done_"),
            data_manager.device_data("_timestep_"),
            force_reset,
        )


class NumbaLogController(CUDALogController):
    """
    Numba Log Controller: manages the Numba logger inside GPU for all the data having
    the flag log_data_across_episode = True.
    The log function will only work for one particular env, even there are multiple
    example_envs running together.

    prerequisite: NumbaFunctionManager is initialized, and the default function list
    has been successfully launched

    Example:
        Please refer to tutorials

    """

    def __init__(self, function_manager: NumbaFunctionManager):
        """
        :param function_manager: NumbaFunctionManager object
        """
        super().__init__(function_manager)

    def _log_one_step(self, data_manager: NumbaDataManager, step: int, env_id: int = 0):
        step = np.int32(step)
        assert env_id < data_manager.meta_info("n_envs")
        env_id = np.int32(env_id)

        for name in data_manager.log_data_list:
            f_shape = data_manager.get_shape(name)
            assert f_shape[0] == data_manager.meta_info(
                "n_envs"
            ), "log function assumes the 0th dimension is n_envs"
            assert f_shape[1] == data_manager.meta_info(
                "n_agents"
            ), "log function assumes the 1st dimension is n_agents"
            if len(f_shape) >= 3:
                if len(f_shape) > 3:
                    raise Exception(
                        "Numba environment.log() temporarily not supports array dimension > 3"
                    )
                feature_dim = np.int32(np.prod(f_shape[2:]))
                data_shape = "is_3d"
            else:
                data_shape = "is_2d"
            dtype = data_manager.get_dtype(name)
            assert "float" in dtype or "int" in dtype, f"unknown dtype: {dtype}"
            if data_shape == "is_3d":
                log_func = self._function_manager.get_function("log_one_step_3d")
                log_func[self._blocks_per_env, self._block](
                    data_manager.device_data(f"{name}_for_log"),
                    data_manager.device_data(name),
                    feature_dim,
                    step,
                    data_manager.meta_info("episode_length"),
                    env_id,
                )
            elif data_shape == "is_2d":
                log_func = self._function_manager.get_function("log_one_step_2d")
                log_func[self._blocks_per_env, self._block](
                    data_manager.device_data(f"{name}_for_log"),
                    data_manager.device_data(name),
                    step,
                    data_manager.meta_info("episode_length"),
                    env_id,
                )

    def _update_log_mask(self, data_manager: NumbaDataManager, step: int):
        """
        Mark the success of the current step and assign 1 for the dense_log_mask,
        update self.last_valid_step
        """
        step = np.int32(step)
        update_mask = self._function_manager.get_function("update_log_mask")
        update_mask[self._blocks_per_env, self._block](
            data_manager.device_data("_log_mask_"),
            step,
            data_manager.meta_info("episode_length"),
        )
        self.last_valid_step = step

    def _reset_log_mask(self, data_manager: NumbaDataManager):
        reset = self._function_manager.get_function("reset_log_mask")
        reset[self._blocks_per_env, self._block](
            data_manager.device_data("_log_mask_"),
            data_manager.meta_info("episode_length"),
        )
