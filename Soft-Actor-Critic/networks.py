"""
Name : networks.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 29.03.2023 10:52
Desc:
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.layers import Activation, Dense, Input
import tensorflow_probability as tfp


class CriticQNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256,
                 name='critic', chkpt_dir='checkpoints/sac'):
        super(CriticQNetwork, self, ).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_action = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')

        # This is the first hidden layer
        self.fc1 = Dense(fc1_dims, activation='relu')
        # This is second hidden layer
        self.fc2 = Dense(fc2_dims, activation='relu')
        # This layer is the action value-function approximation
        # It outputs the estimated action value, for a given state and action
        self.q_function = Dense(1, activation=None)

    # Forward function from pytorch
    def call(self, state, action):
        state_action_pair = tf.concat([state, action], axis=1)

        out1 = self.fc1(state_action_pair)
        out2 = self.fc2(out1)

        # Q(s,a)
        q_value = self.q_function(out2)

        return q_value

# In the first SAC iteration, we use an extra value function network as a baseline.
# theoretically we can infer V by knowing pi and Q, but in practice, it helps stabilize the training.
class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256, name='value', chkpt_dir='checkpoints/sac'):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        # This is the first hidden layer
        self.fc1 = Dense(fc1_dims, activation='relu')
        # This is second hidden layer
        self.fc2 = Dense(fc2_dims, activation='relu')
        # This layer is the state value-function approximation
        # It outputs the estimated state value, for a given state
        self.v_function = Dense(1, activation=None)

    def call(self, state):
        out1 = self.fc1(state)
        out2 = self.fc2(out1)

        # V(s)
        v_value = self.v_function(out2)

        return v_value


class ContinuousActionActorNetwork(keras.Model):
    def __init__(self, max_actions, n_actions, fc1_dims=256, fc2_dims=256, name='actor',  chkpt_dir='checkpoints/sac'):
        super(ContinuousActionActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_actions = max_actions
        self.n_actions = n_actions
        self.noise = 1e-6

        # This is the first hidden layer
        self.fc1 = Dense(fc1_dims, activation='relu')
        # This is second hidden layer
        self.fc2 = Dense(fc2_dims, activation='relu')

        # Needed for continous action space

        # mean
        self.mu = Dense(self.n_actions, activation=None)
        # Standard devitation
        self.sigma = Dense(self.n_actions, activation=None)

    # Not called direclty. Called by get_continous_action
    def call(self, state):
        prob = self.fc1(state)
        prob = self.fc2(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        # play with [self.noise, 1]
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def get_continous_action(self, state, reparameterize_trick=True):
        # reparametrize_trick not implemented

        mean, std = self.call(state)
        normal_dist = tfp.distributions.Normal(mean, std)

        if reparameterize_trick:
            actions = normal_dist.sample()  # + something else if you want to implement
        else:
            actions = normal_dist.sample()

        # dont cutoff action space env.action_space.high
        action = tf.math.tanh(actions) * self.max_actions
        log_probs = normal_dist.log_prob(actions)
        log_probs -= tf.math.log(1 - tf.math.pow(action, 2) + self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

        # mean, log_std = self.forward(state)
        # std = log_std.exp()
        # normal = Normal(mean, std)
        # x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        # action = torch.tanh(x_t)
        # log_prob = normal.log_prob(x_t)
        # # Enforcing Action Bound
        # log_prob -= torch.log(1 - action.pow(2) + epsilon)
        # log_prob = log_prob.sum(1, keepdim=True)
        # return action, log_prob, mean, log_std










        # def forward(self, state):
        #     x = F.relu(self.linear1(state))
        #     x = F.relu(self.linear2(x))
        #     mean = self.mean_linear(x)
        #     log_std = self.log_std_linear(x)
        #     log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        #     return mean, log_std

