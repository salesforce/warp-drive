import numpy as np
from gym import spaces

# seeding code from https://github.com/openai/gym/blob/master/gym/utils/seeding.py
from warp_drive.utils.constants import Constants
from warp_drive.utils.data_feed import DataFeed
from warp_drive.utils.gpu_environment_context import CUDAEnvironmentContext

from single_agent_two_atom.twoatom_actions_base import TwoAtomActionsBase
from single_agent_two_atom.twoatom_actions_3d import TwoAtomActions3D as ActionObj3D

_OBSERVATIONS = Constants.OBSERVATIONS
_ACTIONS = Constants.ACTIONS
_REWARDS = Constants.REWARDS


class SingleAgentTwoAtomChemSearch:

    name = "SingleAgentTwoAtomChemSearch"

    def __init__(self,
                 ienergy=0,
                 max_denergy=0,
                 nx=0,
                 ny=0,
                 nz=0,
                 z_slab_lower=0,
                 z_slab_upper=0,
                 initial_state=None,
                 final_state=None,
                 terminate_reward=10.0,
                 min_reward=-1.0,
                 episode_length=50,
                 en_array=None,
                 env_backend="cpu"):

        self.num_agents = 1

        self.agents = {}
        for agent_id in range(self.num_agents):
            self.agents[agent_id] = True

        assert initial_state is not None
        self.initial_state = np.array(initial_state)

        assert final_state is not None
        self.final_state = np.array(final_state)

        self.norm_distance_a = np.float32(np.linalg.norm(self.final_state[:3] - self.initial_state[:3]))
        self.norm_distance_b = np.float32(np.linalg.norm(self.final_state[3:] - self.initial_state[3:]))
        self.observation_space = None
        self.action_space = None
        self.terminate_reward = terminate_reward
        self.min_reward = min_reward

        assert episode_length > 0
        self.episode_length = episode_length

        action_obj = ActionObj3D(ienergy=ienergy,
                                 max_denergy=max_denergy,
                                 nx=nx,
                                 ny=ny,
                                 nz=nz,
                                 z_slab_lower=z_slab_lower,
                                 z_slab_upper=z_slab_upper,
                                 en_array=en_array)
        self._load_action_class(action_obj)

        self.timestep = None
        self.global_state = self.initial_state

        self.env_backend = env_backend

    def _load_action_class(self, action_obj: TwoAtomActionsBase):
        assert isinstance(action_obj, TwoAtomActionsBase)
        assert action_obj.z_slab_lower <= self.initial_state[2] and action_obj.z_slab_lower <= self.initial_state[5], \
            f"the environment initial state expects to start from the higher position " \
            f"than z_slab_lower as defined in action class "

        self.action_obj = action_obj
        self.action_space = {0: spaces.MultiDiscrete(
            (self.action_obj.atom_action_space,
             self.action_obj.move_action_space
             )
        )
        }
        self.world_dim = np.array([self.action_obj.nx, self.action_obj.ny, self.action_obj.nz] * 2)

    def generate_observation(self):
        x = self.global_state.astype(np.float32) / self.world_dim
        d1 = np.float32(np.linalg.norm(self.global_state[:3] - self.final_state[:3])) / self.norm_distance_a
        d2 = np.float32(np.linalg.norm(self.global_state[3:] - self.final_state[3:])) / self.norm_distance_b
        d = np.array([d1, d2])
        return {0: np.concatenate([x, d])}

    def reset(self):
        self.timestep = 0
        # we still follow the multi-agent schema
        self.global_state = self.initial_state
        return self.generate_observation()

    def is_terminated(self, state):
        return np.all(state == self.final_state).astype(bool)

    def step(self, actions=None):
        self.timestep += 1
        assert isinstance(actions, dict)
        assert len(actions) == self.num_agents

        atom_action, move_action = actions[0][0], actions[0][1]

        old_position = self.global_state
        new_position, reward = self.action_obj.actions[(atom_action, move_action)](old_position)
        self.global_state = new_position

        assert reward is not None
        assert not np.isnan(reward) and not np.isinf(reward)

        reward = np.clip(reward, self.min_reward, 0.0)

        terminated = self.is_terminated(new_position)
        if terminated:
            reward += self.terminate_reward

        rew = {0: reward}
        obs = self.generate_observation()
        done = {"__all__": self.timestep >= self.episode_length or terminated}
        info = {}

        return obs, rew, done, info


class CUDASingleAgentTwoAtomChemSearch(SingleAgentTwoAtomChemSearch, CUDAEnvironmentContext):
    """
    CUDA version of the SingleAgentTwoAtomChemSearch environment.
    """

    def get_data_dictionary(self):
        data_dict = DataFeed()
        data_dict.add_data(
            name="position",
            data=np.atleast_2d(self.initial_state),
            save_copy_and_apply_at_reset=True,
        )

        data_dict.add_data_list(
            [
                ("terminate_reward", self.terminate_reward),
                ("min_reward", self.min_reward),
                ("ienergy", self.action_obj.ienergy),
                ("max_denergy", self.action_obj.max_denergy),
                ("nx", self.action_obj.nx),
                ("ny", self.action_obj.ny),
                ("nz", self.action_obj.nz),
                ("z_slab_lower", self.action_obj.z_slab_lower),
                ("z_slab_upper", self.action_obj.z_slab_upper),
                ("final_state", self.final_state),
                ("norm_distance_a", self.norm_distance_a),
                ("norm_distance_b", self.norm_distance_b),
                ("en_array", self.action_obj.en_array),
            ]
        )
        return data_dict

    def step(self, actions=None):
        self.timestep += 1
        args = [
            "position",
            _ACTIONS,
            "_done_",
            _REWARDS,
            _OBSERVATIONS,
            "terminate_reward",
            "min_reward",
            "ienergy",
            "max_denergy",
            "nx",
            "ny",
            "nz",
            "z_slab_lower",
            "z_slab_upper",
            "final_state",
            "norm_distance_a",
            "norm_distance_b",
            "en_array",
            "_timestep_",
            ("episode_length", "meta"),
        ]
        if self.env_backend == "numba":
            self.cuda_step[
                self.cuda_function_manager.grid, self.cuda_function_manager.block
            ](*self.cuda_step_function_feed(args))
        else:
            raise Exception("CUDASingleAgentTwoAtomChemSearch expects env_backend = 'numba' ")
