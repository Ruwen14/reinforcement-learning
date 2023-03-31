"""
Name : networks.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 29.03.2023 10:34
Desc:
"""

import numpy as np


# Keep track of the last max_size transitions
class ReplayBuffer:
    def __init__(self, replay_size, state_dims,  n_actions):
        self.mem_size = replay_size
        self.mem_cntr = 0
        # [[0,..., state_dims], [0,..., state_dims],..., self.mem_size]
        self.state_memory = np.zeros((self.mem_size, *state_dims))
        self.next_state_memory = np.zeros((self.mem_size, *state_dims))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, terminal):
        # if mem_cntr bigger than mem_size start from 0
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.next_state_memory[index] = next_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def sample_batch(self, batch_size):
        # if we only filled half of replay buffer yet, we only want to sample actual ones, not the zeros we added for allocation
        max_mem = min(self.mem_cntr, self.mem_size)

        # Get batch_size X random indexes between 0 and max_mem
        batch = np.random.choice(max_mem, batch_size)

        states_batch = self.state_memory[batch]
        next_states_batch = self.next_state_memory[batch]
        actions_batch = self.action_memory[batch]
        rewards_batch = self.reward_memory[batch]
        terminals_batch = self.terminal_memory[batch]

        return states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch




