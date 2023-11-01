import unittest

from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from warp_drive.utils.env_registrar import EnvironmentRegistrar
from example_envs.single_agent.classic_control.cartpole.cartpole import \
    ClassicControlCartPoleEnv, CUDAClassicControlCartPoleEnv


env_configs = {
    "test1": {
        "episode_length": 500,
        "reset_pool_size": 0,
        "seed": 32145,
    },
    "test2": {
        "episode_length": 200,
        "reset_pool_size": 0,
        "seed": 54231,
    },
}


class MyTestCase(unittest.TestCase):
    """
    CPU v GPU consistency unit tests
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.testing_class = EnvironmentCPUvsGPU(
            cpu_env_class=ClassicControlCartPoleEnv,
            cuda_env_class=CUDAClassicControlCartPoleEnv,
            env_configs=env_configs,
            gpu_env_backend="numba",
            num_envs=5,
            num_episodes=2,
        )

    def test_env_consistency(self):
        try:
            self.testing_class.test_env_reset_and_step()
        except AssertionError:
            self.fail("ClassicControlCartPoleEnv environment consistency tests failed")
