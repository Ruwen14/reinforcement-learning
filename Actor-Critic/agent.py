import tensorflow as tf
from keras.optimizers import Adam
import tensorflow_probability as tfp
from networks import ActorCriticSharedNetwork
import gym
import numpy as np


# The categorical distribution is a useful tool for modeling
# discrete random variables because it provides a natural way
# to represent a probability distribution over a finite set of discrete values
# and the sum of the probabilities is equal to 1
def categorical_distribution(distribution):
    return tfp.distributions.Categorical(distribution)


class Agent:
    def __init__(self, shared_learning_rate, n_actions, gamma=0.99):
        self.discount_factor = gamma
        self.n_actions = n_actions
        self.shared_learning_rate = shared_learning_rate
        self.last_action_took = None
        # List [0,1,2,...., last action]
        self.action_space = [action for action in range(self.n_actions)]

        # Instance Network
        self.ActorCriticNet = ActorCriticSharedNetwork(self.n_actions)
        self.ActorCriticNet.compile(optimizer=Adam(learning_rate=shared_learning_rate))

    # obervation might only partially describe the state
    # by clearning up observation we get state
    #

    # in the CartPole example the observatoin is
    # [position of cart, velocity of cart, angle of pole, rotation rate of pole]
    def choose_action(self, observation):
        # make observation [4 entries ] to tensor [1,4], adding extra
        # dimension
        state = tf.convert_to_tensor([observation])

        # same uses call() method.
        _, policy_dist = self.ActorCriticNet(state)

        # sample an action
        action_probabilities = categorical_distribution(policy_dist)
        action = action_probabilities.sample()

        self.last_action_took = action

        # [0] for reducing dimension to make it work with tensorflow
        return action.numpy()[0]





    def train_step(self, cur_state, cur_reward, next_state, done):

        # s
        cur_state = tf.convert_to_tensor([cur_state], dtype=tf.float32)
        # s'
        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        # r
        cur_reward = tf.convert_to_tensor(cur_reward, dtype=tf.float32) # no need to add dimension trough [] as not fed to Network

        with tf.GradientTape(persistent=True) as tape:

            # Estimate the current state_value of the state and give action
            # V(s) and a ~ pi(a|s)
            cur_state_value, policy_dist = self.ActorCriticNet(cur_state)
            action = self.last_action_took

            # Estimate the state_value of the next state
            # V(s')
            next_state_value, _ = self.ActorCriticNet(next_state)

            # Get rid of extra dimension
            cur_state_value = tf.squeeze(cur_state_value)
            next_state_value = tf.squeeze(next_state_value)


            action_probabilities = categorical_distribution(policy_dist)

            #In summary, the log probability in an Actor-Critic algorithm
            # is a representation of the probability of taking a particular
            # action given the current state of the environment
            # When a low log probability of an action occurs in an
            # Actor-Critic reinforcement learning (RL) algorithm,
            # the policy gradient will be small, indicating that
            # the policy network should be updated only slightly
            # in response to the received reward.
            log_prop = action_probabilities.log_prob(self.last_action_took)

            # Boostrapped Target value
            # 1-int(done) in case of terminal state, as the terminal state value is 0, as no rewards follow the terminal state
            target_value = cur_reward + self.discount_factor * next_state_value*(1-int(done))

            # Temporal Difference is also used to approximate the advantage Value
            delta_temporal_difference = target_value - cur_state_value



            # Equivalent to log pi(a|s) * A(s,a), where A(s,a) (advantage) is approximated by Temporal Diffence
            actor_loss = -log_prop * delta_temporal_difference

            # Basically the mean squarred error between the target_value and our state_value
            critic_loss = delta_temporal_difference**2

            total_loss = actor_loss + critic_loss

        # gradient of the loss with respect to the parameters.
        gradient = tape.gradient(total_loss, self.ActorCriticNet.trainable_variables)
        self.ActorCriticNet.optimizer.apply_gradients(zip(
                gradient, self.ActorCriticNet.trainable_variables)
        )

