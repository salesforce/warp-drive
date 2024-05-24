import logging
import os
import sys

import torch

from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar

from rlutils.common import get_project_root, load_en_array
from single_agent_one_atom.single_agent_one_atom import SingleAgentOneAtomChemSearch, CUDASingleAgentOneAtomChemSearch


logging.getLogger().setLevel(logging.ERROR)

_NUM_GPUS_AVAILABLE = torch.cuda.device_count()
assert _NUM_GPUS_AVAILABLE > 0, "This script needs a GPU to run!"

env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(CUDASingleAgentOneAtomChemSearch.name,
                                    "single_agent_one_atom.single_agent_one_atom_step_numba",
                                    env_backend="numba")

en_array_diffusion = load_en_array(f"{get_project_root()}/en_array/en_array_diffusion.npy")
en_array_double_grids = load_en_array(f"{get_project_root()}/en_array/en_array_double_grids.npy")

env_configs = {
    "test1": {
        "ienergy": -193.6023,
        "max_denergy": 20,
        "nx": 20,
        "ny": 18,
        "nz": 100,
        "z_slab_lower": 58,
        "z_slab_upper": 68,
        "initial_state": [5, 9, 60],
        "final_state": [10, 0, 60],
        "terminate_reward": 10.0,
        "episode_length": 50,
        "en_array": en_array_diffusion,
    },
    "test2": {
        "ienergy": -193.6023,
        "max_denergy": 20,
        "nx": 20,
        "ny": 18,
        "nz": 100,
        "z_slab_lower": 58,
        "z_slab_upper": 68,
        "initial_state": [5, 9, 61],
        "final_state": [10, 0, 60],
        "terminate_reward": 10.0,
        "min_reward": -0.2,
        "episode_length": 50,
        "en_array": en_array_diffusion,
    },
    "test3": {
        "ienergy": -227.689938,
        "max_denergy": 20.0,
        "nx": 39,
        "ny": 35,
        "nz": 99,
        "z_slab_lower": 56,
        "z_slab_upper": 95,
        "initial_state": [6, 24, 90],
        "final_state": [10, 32, 76],
        "terminate_reward": 30.0,
        "episode_length": 50,
        "en_array": en_array_double_grids,
    },

}
testing_class = EnvironmentCPUvsGPU(
            cpu_env_class=SingleAgentOneAtomChemSearch,
            cuda_env_class=CUDASingleAgentOneAtomChemSearch,
            env_configs=env_configs,
            gpu_env_backend="numba",
            num_envs=5,
            num_episodes=2,
            env_registrar=env_registrar,
        )

testing_class.test_env_reset_and_step(consistency_threshold_pct=1, seed=17)