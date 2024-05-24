import numpy as np


class OneAtomActionsBase(object):

    def __init__(self, ienergy, max_denergy, nx, ny, nz, z_slab_lower, z_slab_upper, en_array):

        self.ienergy = ienergy  # Given the reference energy to make negative reward
        self.max_denergy = max_denergy
        self.epsilon = 1e-8 

        self.nx, self.ny, self.nz = nx, ny, nz  # grid defined for Lammps

        self.z_slab_lower = z_slab_lower
        self.z_slab_upper = z_slab_upper
        effective_z = self.z_slab_upper - self.z_slab_lower

        en_array_shape = (self.nx, self.ny, effective_z)

        self.en_array = en_array
        assert self.en_array.shape == en_array_shape

        self.actions = None

    def is_bad_state(self, state):
        atoms_beyond_valid_region = state[2] < self.z_slab_lower or \
                                    state[2] >= self.z_slab_upper

        return atoms_beyond_valid_region

