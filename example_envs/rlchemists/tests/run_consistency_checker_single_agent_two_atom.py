import logging
import os
import sys

import torch

from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar

from rlutils.common import get_project_root, load_en_array
from single_agent_two_atom.single_agent_two_atom import SingleAgentTwoAtomChemSearch, CUDASingleAgentTwoAtomChemSearch


logging.getLogger().setLevel(logging.ERROR)

_NUM_GPUS_AVAILABLE = torch.cuda.device_count()
assert _NUM_GPUS_AVAILABLE > 0, "This script needs a GPU to run!"

env_registrar = EnvironmentRegistrar()
env_registrar.add_cuda_env_src_path(CUDASingleAgentTwoAtomChemSearch.name,
                                    "single_agent_two_atom.single_agent_two_atom_step_numba",
                                    env_backend="numba")

en_array = load_en_array(f"{get_project_root()}/en_array/en_array_2atom_diffusion.npy")

env_configs = {
    "test1": {
        "ienergy": -200.780,
        "max_denergy": 20,
        "nx": 8,
        "ny": 8,
        "nz": 15,
        "z_slab_lower": 9,
        "z_slab_upper": 14,
        "initial_state": [4, 4, 10, 4, 4, 11],
        "final_state": [0, 4, 10, 0, 4, 11],
        "terminate_reward": 30.0,
        "episode_length": 50,
        "en_array": en_array,
    },
    "test2": {
        "ienergy": -200.780,
        "max_denergy": 20,
        "nx": 8,
        "ny": 8,
        "nz": 15,
        "z_slab_lower": 9,
        "z_slab_upper": 14,
        "initial_state": [4, 3, 12, 4, 7, 9],
        "final_state": [6, 4, 10, 0, 4, 11],
        "terminate_reward": 30.0,
        "episode_length": 50,
        "en_array": en_array,
    },
    "test3": {
        "ienergy": -200.780,
        "max_denergy": 20,
        "nx": 8,
        "ny": 8,
        "nz": 15,
        "z_slab_lower": 9,
        "z_slab_upper": 14,
        "initial_state": [4, 3, 12, 4, 7, 9],
        "final_state": [4, 4, 12, 4, 7, 10],
        "terminate_reward": 30.0,
        "episode_length": 50,
        "en_array": en_array,
    },
}
testing_class = EnvironmentCPUvsGPU(
            cpu_env_class=SingleAgentTwoAtomChemSearch,
            cuda_env_class=CUDASingleAgentTwoAtomChemSearch,
            env_configs=env_configs,
            gpu_env_backend="numba",
            num_envs=5,
            num_episodes=2,
            env_registrar=env_registrar,
        )

testing_class.test_env_reset_and_step(consistency_threshold_pct=1, seed=17)