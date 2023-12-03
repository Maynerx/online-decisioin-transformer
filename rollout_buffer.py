import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialisation des buffers pour les états, actions, récompenses, etc.
        self.states = [0]*self.buffer_size
        self.actions = [0]*self.buffer_size
        self.rewards = [0]*self.buffer_size
        self.next_states = [0]*self.buffer_size
        self.dones = [0]*self.buffer_size
        self.rtg = [0]*self.buffer_size
        self.timestep = [0]*self.buffer_size

        # Indice du dernier élément dans le buffer
        self.index = 0
        self.full = False

    def add_experience(self, state, action, reward, next_state, done, rtg, timestep):
        self.states[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.next_states[self.index] = next_state
        self.dones[self.index] = done
        self.rtg[self.index] = rtg
        self.timestep[self.index] = timestep

        # Met à jour l'indice et le drapeau indiquant si le buffer est plein
        self.index = (self.index + 1) % self.buffer_size
        if self.index == 0:
            self.full = True

    def __get_sample__(self, indice):
        batch = {
            'states': self.states[indice],
            'actions': self.actions[indice],
            'rewards': self.rewards[indice],
            'next_states': self.next_states[indice],
            'dones': self.dones[indice],
            'rtg' : self.rtg[indice],
            'timestep' : self.timestep[indice]
        }
        return batch
    
    def __generate__(self, batch_size):
        arr = []
        while len(arr) < batch_size:
            n = np.random.randint(0, self.index if not self.full else self.buffer_size)
            if not n in arr:
                arr.append(n)
        return arr

    def get_batch(self, batch_size):
        batchs = []
        # Échantillonne aléatoirement un lot de données à partir du buffer
        if not self.full and self.index < batch_size:
            indices = range(self.index)
        else:
            indices = self.__generate__(batch_size)
        for i in indices: 
            batchs.append(self.__get_sample__(i))
        return batchs