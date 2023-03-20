"""
Name : networks.py.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 14.03.2023 15:57
Desc:
"""

from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import tensorflow as tf
import sys

# weird fix for https://stackoverflow.com/questions/65366442/cannot-convert-a-symbolic-keras-input-output-to-a-numpy-array-typeerror-when-usi
# only for version by video https://www.youtube.com/watch?v=2vJtbAha3To&t=770s
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class ActorCriticSharedNetwork(object):
    def __init__(self,  alpha, beta, n_actions=4, fc1_dims=1024, fc2_dims=512, state_dims=8):

        self.state_dims = state_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.alpha = alpha
        self.beta = beta

    def model(self):
        input = Input(shape=(self.state_dims,))
        delta = Input(shape=[1])

        # We share a common layer for both Actor and Critic.
        # As its input it takes the state. The State dimensionality
        # is affected by its represantation.

        # This is the first hidden layer
        fc1 = Dense(self.fc1_dims, activation='relu')(input)

        #This is the second hidden layer
        fc2 = Dense(self.fc2_dims, activation='relu')(fc1)

        # [CRITIC]
        # This layer is the state-value-function approximation
        # It outputs the estimated state-value, for a given state
        value_function = Dense(1, activation='linear')(fc2)

        # [ACTOR]
        # This layer is the for the policy.
        # Given a state, it outputs the probability distribution
        # over the n- output actions (n_actions)
        # softmax needed to normalize distribution from 0.0 to 1.0
        policy = Dense(self.n_actions, activation='softmax')(fc2)


        # Custom Actor loss function
        # y_true is one hot repr of action took [0,1,0,0]
        def custom_loss(y_true, y_pred):
            # so we dont log of zero which is inf
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_lik = y_true*K.log(out)
            return K.sum(-log_lik*delta)

        actor = Model(inputs=[input, delta], outputs=[policy])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[input], outputs=[value_function])
        critic.compile(optimizer=Adam(lr=self.beta), loss='mean_squared_error')

        # To just calculate the feed forward on the network, dont want to use for training
        policy_ = Model(inputs=[input], outputs=[policy])

        return actor, critic, policy_



