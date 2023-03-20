"""
Name : agent.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 14.03.2023 15:30
Desc:
"""

from keras import backend as K
from keras.layers import Activation, Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from networks import ActorCriticSharedNetwork

class Agent(object):
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4,layer1_size=1024, layer2_size=512, state_dims=8):
        self.discount_factor = gamma
        self.alpha = alpha
        self.beta = beta
        self.state_dims = state_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions

        self.actor, self.critic, self.policy = ActorCriticSharedNetwork(self.alpha, self.beta, self.n_actions, self.fc1_dims, self.fc2_dims, self.state_dims).model()

        # [0,1,2,3]
        self.action_space = [action for action in range(self.n_actions)]

    def choose_action(self, observation):
        # adding extra dimension
        state = observation[np.newaxis, :]

        # sample an action according to our distribution
        action_probabilities = self.policy.predict(state)[0]
        action = np.random.choice(a=self.action_space, p=action_probabilities)

        return action


    def train_step(self, state, reward, next_state, action, done):
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        # Estimate the current state_value of the state
        # V(s)
        state_value = self.critic.predict(state)

        # Estimate the state_value of the next state
        # V(s')
        next_state_value = self.critic.predict(next_state)

        # Boostrapped Target value
        # 1-int(done) in case of terminal state, as the terminal state value is 0, as no rewards follow the terminal state
        target_value = reward + self.discount_factor * next_state_value * (1 - int(done))

        # Compute the correction (TD Error)  $\delta_t$ for state-value prediction at time $t$:
        # Temporal Difference is also used to approximate the advantage Value
        delta_temporal_difference = target_value - state_value

        # make action into 3 -> [[0. 0. 0. 1.]] (one hot encoding)
        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0


        self.actor.fit([state, delta_temporal_difference], actions, verbose=0)
        self.critic.fit(state, target_value, verbose=0)



        #     action = self.last_action_took
        #
        #     # Estimate the state_value of the next state
        #     # V(s')
        #     next_state_value, _ = self.ActorCriticNet(next_state)
        #
        #     # Get rid of extra dimension
        #     cur_state_value = tf.squeeze(cur_state_value)
        #     next_state_value = tf.squeeze(next_state_value)
        #
        #
        #     action_probabilities = categorical_distribution(policy_dist)
        #
        #     #In summary, the log probability in an One-Step-TD-Actor-Critic algorithm
        #     # is a representation of the probability of taking a particular
        #     # action given the current state of the environment
        #     # When a low log probability of an action occurs in an
        #     # One-Step-TD-Actor-Critic reinforcement learning (RL) algorithm,
        #     # the policy gradient will be small, indicating that
        #     # the policy network should be updated only slightly
        #     # in response to the received reward.
        #     log_prop = action_probabilities.log_prob(self.last_action_took)
        #
        #     # Boostrapped Target value
        #     # 1-int(done) in case of terminal state, as the terminal state value is 0, as no rewards follow the terminal state
        #     target_value = cur_reward + self.discount_factor * next_state_value*(1-int(done))
        #
        #     # Compute the correction (TD Error)  $\delta_t$ for state-value prediction at time $t$:
        #     # Temporal Difference is also used to approximate the advantage Value
        #     delta_temporal_difference = target_value - cur_state_value
        #
        #     # Get feedback for action a by calculation advantage according to critics estimation of action
        #     # Estimated by TD Error
        #     advantage = delta_temporal_difference
        #
        #     # adjust policy according to advantage
        #     # Equivalent to log pi(a|s) * A(s,a), where A(s,a) (advantage) is approximated by Temporal Diffence
        #     actor_loss = -log_prop * delta_temporal_difference
        #
        #     # adjust state-value /advantage-predicition according to prediction error TD_Error
        #     # Basically the mean squarred error between the target_value and our state_value
        #     critic_loss = delta_temporal_difference**2
        #
        #     total_loss = actor_loss + critic_loss
        #
        # # gradient of the loss with respect to the parameters.
        # # 1) Build gradient in the direction proportional to the advantage you can get, to encourage/discourage action a,
        # #   by increasing/decreasing probability in policy
        # #2) To reduce TD error, we square it and minimize it with gradient descent according to w
        # gradient = tape.gradient(total_loss, self.ActorCriticNet.trainable_variables)
        # self.ActorCriticNet.optimizer.apply_gradients(zip(
        #         gradient, self.ActorCriticNet.trainable_variables)
        # )
        #


