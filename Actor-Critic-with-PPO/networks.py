"""
Name : networks.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 24.03.2023 15:45
Desc:
"""


import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


# In the standard A3C algorithm, the value function is estimated using a one-step lookahead,
# meaning that it is estimated as the immediate reward plus the estimated value of the next state.
# However, in an n-step version of A3C, we save our transitions in form of a T_transitions-step big Batch, with  regard
# to, we do our update.


# PPO runs the policy using T_transitions parallel actors each collecting data,
# and then it samples mini-batches of this data to train for K epochs using the Clipped Surrogate Objective function. See full algorithm below


# he basic idea is that you have T_transitions workers, each with an environment. After all the environments have been stepped for timesteps_per_actorbatch
#  steps, all of the experience is put into a dataset, and the network is trained on the dataset using mini-batch sizes determined by optim_batchsize
# .
# use ppo1 and ppo2 docs from baseline for comments.

# In PPO each Agent runs steps trough the environment for Transition-steps T"
# After T steps all the transitions are syncronized to this RolloutBuffer
# In a Multi-Agent setup like A2C/A3C this Buffer has the size of "Transition-Steps T" * "parallel Agents/Envs"
# As this example only uses One Agent for the sake of simplicity, the size of this Buffer is simply "Transition-Steps T"
# The Transitions in this buffer are used to update the Actor-Critic-Network using mini-batches
class  RolloutBatchBufferMaybeFedbyMultipleAgents(object):
    def __init__(self, minibatch_size):
        self.rewards = []
        self.actions = []
        self.states = []
        self.state_values = []
        self.dones = []
        self.probs = []
        self.minibatch_size = minibatch_size

    def generate_indices_of_minibatches_from_rollouts(self):
        """
        Creates randomly shuffled indices of to our T_transitions transitions and breakts it into T/minibatch_size number of
        minibatch indices.

        In other words, it breaks our collected transitions into T_transitions minibatches of size minibatch_size and returns the indices.

        Ensures that all samples are used for one Epoch

        For example for 2048 Transitions and minibatch_size of 64 we would create 32 minibatch indices

        for epochs(10):
            randomize batch
            for minibatch in minibatches:
                update with minibatch

        """

        len_transitions = len(self.states)
        batch_start = np.arange(0, len_transitions, self.minibatch_size)
        indices = np.arange(len_transitions, dtype=np.int64)
        np.random.shuffle(indices)
        minibatch_indices = [indices[i:i + self.minibatch_size] for i in batch_start]

        return minibatch_indices


    def rollouts(self):
            return {
            "states" : np.array(self.states),
            "actions" : np.array(self.actions),
            "log_probs" : np.array(self.probs),
            "state_values" : np.array(self.state_values),
            "rewards" : np.array(self.rewards),
            "terminals" : np.array(self.dones)
        }


    def append_transition(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.state_values.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_all(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.state_values = []



class ActorNetwork(nn.Module):
    def __init__(self, n_actions, state_dims, alpha,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='./checkpoints'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        # ACTOR NETWORK
        # ------------------------------------------------------------------------------#
        # Dont use shared parameters for Actor/Critic, as it complicates loss function
        # for examle shown by "Machine Leanring with Phil".

        self.actor = nn.Sequential(
            nn.Linear(*state_dims, fc1_dims), # Input 1
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims), # Hidden1
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions), # Hidden2
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        policy = self.actor(state)

        return policy

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))



class CriticNetwork(nn.Module):
    def __init__(self, state_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='./checkpoints'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        # Critic NETWORK
        # ------------------------------------------------------------------------------#
        # Dont use shared parameters for Actor/Critic, as it complicates loss function
        # for examle shown by "Machine Leanring with Phil".

        self.critic = nn.Sequential(
                nn.Linear(*state_dims, fc1_dims), # Input
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims), # Hidden  1
                nn.ReLU(),
                nn.Linear(fc2_dims, 1) # Hidden 2
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))







