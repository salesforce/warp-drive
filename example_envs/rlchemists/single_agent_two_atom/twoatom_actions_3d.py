from single_agent_two_atom.twoatom_actions_base import TwoAtomActionsBase


class TwoAtomActions3D(TwoAtomActionsBase):

    def __init__(self,
                 ienergy=0,
                 max_denergy=0,
                 nx=0,
                 ny=0,
                 nz=0,
                 z_slab_lower=0,
                 z_slab_upper=0,
                 en_array=None):

        super().__init__(ienergy, max_denergy, nx, ny, nz, z_slab_lower, z_slab_upper, en_array)

        self.actions = {
            # action for atom 1
            (0, 0): self._action_0,
            (0, 1): self._action_1,
            (0, 2): self._action_2,
            (0, 3): self._action_3,
            (0, 4): self._action_4,
            (0, 5): self._action_5,
            # action for atom 2
            (1, 0): self._action_6,
            (1, 1): self._action_7,
            (1, 2): self._action_8,
            (1, 3): self._action_9,
            (1, 4): self._action_10,
            (1, 5): self._action_11,
        }
        self.atom_action_space = 2 # which atom to move
        self.move_action_space = int(len(self.actions) / self.atom_action_space) # where to move, equal to 6

        # self.action_size = int(len(self.actions))

    def calculate_denergy(self, state):
        denergy = self.ienergy - self.en_array[state[0]][state[1]][state[2] - self.z_slab_lower][state[3]][state[4]][
            state[5] - self.z_slab_lower]
        return denergy

    def _action_0(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[0] += 1 

        if state[0] < 0:
           state[0] += self.nx

        if state[0] >= self.nx: 
           state[0] -= self.nx 

        denergy = self.calculate_denergy(state)
    
        return state, denergy / self.max_denergy

    def _action_1(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[0] -= 1 

        if state[0] < 0:
           state[0] += self.nx 

        if state[0] >= self.nx: 
           state[0] -= self.nx

        denergy = self.calculate_denergy(state)
    
        return state, denergy / self.max_denergy

    def _action_2(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[1] += 1 

        if state[1] < 0:
           state[1] += self.ny 

        if state[1] >= self.ny: 
           state[1] -= self.ny 

        denergy = self.calculate_denergy(state)
    
        return state, denergy / self.max_denergy

    def _action_3(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[1] -= 1

        if state[1] < 0:
           state[1] += self.ny 

        if state[1] >= self.ny: 
           state[1] -= self.ny 

        denergy = self.calculate_denergy(state)
    
        return state, denergy / self.max_denergy

    def _action_4(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[2] += 1

        if state[2] < 0:
           state[2] += self.nz 

        if state[2] >= self.nz: 
           state[2] -= self.nz

        if self.is_bad_state(state):
            state[2] -= 1
            denergy = -self.max_denergy
        else:
            denergy = self.calculate_denergy(state)
    
        return state, denergy / self.max_denergy

    def _action_5(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[2] -= 1

        if state[2] < 0:
           state[2] += self.nz 

        if state[2] >= self.nz: 
           state[2] -= self.nz

        if self.is_bad_state(state):
            state[2] += 1
            denergy = -self.max_denergy
        else:
            denergy = self.calculate_denergy(state)
    
        return state, denergy / self.max_denergy

    def _action_6(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[3] += 1

        if state[3] < 0:
            state[3] += self.nx

        if state[3] >= self.nx:
            state[3] -= self.nx

        denergy = self.calculate_denergy(state)

        return state, denergy / self.max_denergy

    def _action_7(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[3] -= 1

        if state[3] < 0:
            state[3] += self.nx

        if state[3] >= self.nx:
            state[3] -= self.nx

        denergy = self.calculate_denergy(state)

        return state, denergy / self.max_denergy

    def _action_8(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[4] += 1

        if state[4] < 0:
            state[4] += self.ny

        if state[4] >= self.ny:
            state[4] -= self.ny

        denergy = self.calculate_denergy(state)

        return state, denergy / self.max_denergy

    def _action_9(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[4] -= 1

        if state[4] < 0:
            state[4] += self.ny

        if state[4] >= self.ny:
            state[4] -= self.ny

        denergy = self.calculate_denergy(state)

        return state, denergy / self.max_denergy

    def _action_10(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[5] += 1

        if state[5] < 0:
            state[5] += self.nz

        if state[5] >= self.nz:
            state[5] -= self.nz

        if self.is_bad_state(state):
            state[5] -= 1
            denergy = -self.max_denergy
        else:
            denergy = self.calculate_denergy(state)

        return state, denergy / self.max_denergy

    def _action_11(self, state):
        """
        :param state: old_state
        :return: new_state and reward
        """
        state = state.copy()

        state[5] -= 1

        if state[5] < 0:
            state[5] += self.nz

        if state[5] >= self.nz:
            state[5] -= self.nz

        if self.is_bad_state(state):
            state[5] += 1
            denergy = -self.max_denergy
        else:
            denergy = self.calculate_denergy(state)

        return state, denergy / self.max_denergy







