import unittest
import numpy as np
import torch

from warp_drive.env_cpu_gpu_consistency_checker import EnvironmentCPUvsGPU
from example_envs.single_agent.classic_control.mountain_car.mountain_car import \
    ClassicControlMountainCarEnv, CUDAClassicControlMountainCarEnv
from warp_drive.env_wrapper import EnvWrapper


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
            cpu_env_class=ClassicControlMountainCarEnv,
            cuda_env_class=CUDAClassicControlMountainCarEnv,
            env_configs=env_configs,
            gpu_env_backend="numba",
            num_envs=5,
            num_episodes=2,
        )

    def test_env_consistency(self):
        try:
            self.testing_class.test_env_reset_and_step()
        except AssertionError:
            self.fail("ClassicControlMountainCarEnv environment consistency tests failed")

    def test_reset_pool(self):
        env_wrapper = EnvWrapper(
            env_obj=CUDAClassicControlMountainCarEnv(episode_length=100, reset_pool_size=3),
            num_envs=3,
            env_backend="numba",
        )
        env_wrapper.reset_all_envs()
        env_wrapper.env_resetter.init_reset_pool(env_wrapper.cuda_data_manager, seed=12345)
        self.assertTrue(env_wrapper.cuda_data_manager.reset_target_to_pool["state"] == "state_reset_pool")

        # squeeze() the agent dimension which is 1 always
        state_after_initial_reset = env_wrapper.cuda_data_manager.pull_data_from_device("state").squeeze()

        reset_pool = env_wrapper.cuda_data_manager.pull_data_from_device(
            env_wrapper.cuda_data_manager.get_reset_pool("state"))
        reset_pool_mean = reset_pool.mean(axis=0).squeeze()

        # we only need to check the 0th element of state because state[1] = 0 for reset always
        self.assertTrue(reset_pool.std(axis=0).squeeze()[0] > 1e-4)

        env_wrapper.cuda_data_manager.data_on_device_via_torch("_done_")[:] = torch.from_numpy(
            np.array([1, 1, 0])
        ).cuda()

        state_values = {0: [], 1: [], 2: []}
        for _ in range(10000):
            env_wrapper.env_resetter.reset_when_done(env_wrapper.cuda_data_manager, mode="if_done", undo_done_after_reset=False)
            res = env_wrapper.cuda_data_manager.pull_data_from_device("state")
            state_values[0].append(res[0])
            state_values[1].append(res[1])
            state_values[2].append(res[2])

        state_values_env0_mean = np.stack(state_values[0]).mean(axis=0).squeeze()
        state_values_env1_mean = np.stack(state_values[1]).mean(axis=0).squeeze()
        state_values_env2_mean = np.stack(state_values[2]).mean(axis=0).squeeze()

        self.assertTrue(np.absolute(state_values_env0_mean[0] - reset_pool_mean[0]) < 0.1 * abs(reset_pool_mean[0]))
        self.assertTrue(np.absolute(state_values_env1_mean[0] - reset_pool_mean[0]) < 0.1 * abs(reset_pool_mean[0]))
        self.assertTrue(
            np.absolute(
                state_values_env2_mean[0] - state_after_initial_reset[0][0]
                        ) < 0.001 * abs(state_after_initial_reset[0][0])
            )


