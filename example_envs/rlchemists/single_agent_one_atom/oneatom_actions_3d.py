from single_agent_one_atom.oneatom_actions_base import OneAtomActionsBase


class OneAtomActions3D(OneAtomActionsBase):

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
            # action for atom
            0: self._action_0,
            1: self._action_1,
            2: self._action_2,
            3: self._action_3,
            4: self._action_4,
            5: self._action_5,
        }

        self.action_size = int(len(self.actions))

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

        denergy = self.ienergy - self.en_array[state[0]][state[1]][state[2]-self.z_slab_lower]
    
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

        denergy = self.ienergy - self.en_array[state[0]][state[1]][state[2]-self.z_slab_lower]
    
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

        denergy = self.ienergy - self.en_array[state[0]][state[1]][state[2]-self.z_slab_lower]
    
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

        denergy = self.ienergy - self.en_array[state[0]][state[1]][state[2]-self.z_slab_lower]
    
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

        # we do no penalize the bad state otherwise it gives a huge bias for z-direction exploration
        # instead, we cancel the action
        if self.is_bad_state(state):
            state[2] -= 1
            denergy = -self.max_denergy
        else:
            denergy = self.ienergy - self.en_array[state[0]][state[1]][state[2]-self.z_slab_lower]
    
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

        # we do no penalize the bad state otherwise it gives a huge bias for z-direction exploration
        # instead, we cancel the action
        if self.is_bad_state(state):
            state[2] += 1
            denergy = -self.max_denergy
        else:
            denergy = self.ienergy - self.en_array[state[0]][state[1]][state[2]-self.z_slab_lower]
    
        return state, denergy / self.max_denergy
