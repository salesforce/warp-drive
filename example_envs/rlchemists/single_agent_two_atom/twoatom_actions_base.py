import numpy as np


class TwoAtomActionsBase(object):

    def __init__(self, ienergy, max_denergy, nx, ny, nz, z_slab_lower, z_slab_upper, en_array):

        self.ienergy = ienergy  # Given the reference energy to make negative reward
        self.max_denergy = max_denergy
        self.epsilon = 1e-8 

        self.nx, self.ny, self.nz = nx, ny, nz  # grid defined for Lammps

        self.z_slab_lower = z_slab_lower
        self.z_slab_upper = z_slab_upper
        effective_z = self.z_slab_upper - self.z_slab_lower

        en_array_shape = (self.nx, self.ny, effective_z, self.nx, self.ny, effective_z)

        self.en_array = en_array
        assert self.en_array.shape == en_array_shape, \
            f"en_array has the shape: {self.en_array.shape} " \
            f"while the input configuration has the shape {en_array_shape}"

        self.actions = None

        self.atom_action_space = None
        self.move_action_space = None

    def is_bad_state(self, state):
        atoms_beyond_valid_region = state[2] < self.z_slab_lower or \
                                    state[2] >= self.z_slab_upper or \
                                    state[5] < self.z_slab_lower or \
                                    state[5] >= self.z_slab_upper
        # atoms_overlap = (state[3] - state[0]) ** 2 + \
        #                 (state[4] - state[1]) ** 2 + \
        #                 (state[5] - state[2]) ** 2 <= 8

        return atoms_beyond_valid_region

