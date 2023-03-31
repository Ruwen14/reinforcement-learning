"""
Name : agent.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 29.03.2023 12:08
Desc:
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from ReplayBuffer import ReplayBuffer
from networks import ContinuousActionActorNetwork, CriticQNetwork, ValueNetwork

class Agent(object):
    def __init__(self, state_dims, env, n_actions, alpha=0.0003, beta=0.0003,
            gamma=0.99, replay_size=1000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=256, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = ReplayBuffer(replay_size, state_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        # maximum value of continous action .high mapped to 1
        self.actor = ContinuousActionActorNetwork(n_actions=n_actions, name='actor', max_actions=env.action_space.high)

        self.critic_1 = CriticQNetwork(n_actions=n_actions, name='critic_1')
        self.critic_2 = CriticQNetwork(n_actions=n_actions, name='critic_2')


        self.soft_value_func = ValueNetwork(name='value')
        self.soft_value_target_func = ValueNetwork(name='target_value')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))

        self.soft_value_func.compile(optimizer=Adam(learning_rate=beta))
        # Theroretically not needed as we dont make gradients on it, because we copy gradients from value function delayed
        self.soft_value_target_func.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.hard_target_update()

    def choose_action(self, state):
        state = tf.convert_to_tensor([state])
        actions,_ = self.actor.get_continous_action(state)

        # get dims away
        return actions[0]

    def learn(self):
        # Not enough samples
        if self.replay_buffer.mem_cntr < self.batch_size:
            return

        # Sample batch of size 'self.batch_size' from replay buffer
        states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch = self.replay_buffer.sample_batch(self.batch_size)

        states_batch = tf.convert_to_tensor(states_batch, dtype=tf.float32)
        next_states_batch = tf.convert_to_tensor(next_states_batch, dtype=tf.float32)
        rewards_batch = tf.convert_to_tensor(rewards_batch, dtype=tf.float32)
        actions_batch = tf.convert_to_tensor(actions_batch, dtype=tf.float32)

        # Update Soft-Value-Function
        # ************************************************************************************************************
        with tf.GradientTape() as tape:
            # Sample Actions according to current policy
            current_policy_actions, log_probs = self.actor.get_continous_action(states_batch)
            log_probs = tf.squeeze(log_probs,1)

            # we need to sample Q according to actions from current policy, as otherwise we cannot use it as target.
            # 𝑄𝜃_1(𝑠,𝑎)
            critic_q1 = self.critic_1(states_batch, current_policy_actions)
            # 𝑄𝜃_2(𝑠,𝑎)
            critic_q2 = self.critic_1(states_batch, current_policy_actions)
            # min(𝑄𝜃_1(𝑠,𝑎), 𝑄𝜃_2(𝑠,𝑎)), to reduce bias overestimation
            critic_q_min = tf.squeeze(tf.math.minimum(critic_q1, critic_q2), 1)


            # Target for addition soft-value-function, we can use Q, as the actions are
            # sampled according to current policy, instead of the replay buffer ("SAC V1")
            # Vsoft_target=𝔼_a∼𝜋(𝑎|𝑠)[𝑄𝜃min(𝑠,𝑎)−𝛼log(𝜋(𝑎|𝑠))]
            soft_target_values = critic_q_min - log_probs

            # 𝑉̂_pred
            soft_pred_values = tf.squeeze(self.soft_value_func(states_batch), 1)



            # 𝓛_Vsoft = 1/N(Vsoft_target-𝑉̂_pred)^2
            soft_value_func_loss = 0.5 * keras.losses.MSE(soft_pred_values, soft_target_values)

        value_network_gradient = tape.gradient(soft_value_func_loss, self.soft_value_func.trainable_variables)
        self.soft_value_func.optimizer.apply_gradients(zip(value_network_gradient, self.soft_value_func.trainable_variables))
        # ************************************************************************************************************

        # Update Actor
        # ************************************************************************************************************
        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            current_policy_actions, log_probs = self.actor.get_continous_action(states_batch, reparameterize_trick=True)
            log_probs = tf.squeeze(log_probs,1)

            # we need to sample Q according to actions from current policy, as otherwise we cannot use it in the objective function.
            # 𝑄𝜃_1(𝑠,𝑎)
            critic_q1 = self.critic_1(states_batch, current_policy_actions)
            # 𝑄𝜃_2(𝑠,𝑎)
            critic_q2 = self.critic_1(states_batch, current_policy_actions)
            # min(𝑄𝜃_1(𝑠,𝑎), 𝑄𝜃_2(𝑠,𝑎)), to reduce bias overestimation
            critic_q_min = tf.squeeze(tf.math.minimum(critic_q1, critic_q2), 1)

            # Actor loss is formulated so to maximize the likelyhood of actions a∼𝜋(𝑎|𝑠) (current policy),
            # that would result in high Q-Value estimates. Additionaly, the policy encourages the policy
            # to maintain its entropy high enough to help explore, discover, and capture multi-modal
            # optimal policies.
            # Loss can be captures as surrogate to J(π) = V_soft(s), which substitutes Q-value from our
            # Critic-Q-Network. We need to sample actions according to current policy, instead of replay
            # buffer so we can use Q-value
            # 𝓛_actor=𝔼_a∼𝜋(𝑎|𝑠)[𝑄𝜃min(𝑠,𝑎)−𝛼log(𝜋(𝑎|𝑠))]
            actor_loss = log_probs - critic_q_min
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        # ************************************************************************************************************


        # Update Double-Critic-Q-Networks
        # ************************************************************************************************************
        # Remember we use 2 Critic-Q-Networks to reduce overestimation bias.
        # Therefore  we need to compute the losses seperately and make gradient update on them.
        with tf.GradientTape(persistent=True) as tape:
            # V_soft(sₜ₊₁)
            soft_target_values_next= tf.squeeze(self.soft_value_target_func(next_states_batch), 1)

            # 𝑄𝜃_1(𝑠,𝑎)
            critic_q1_on_batch = tf.squeeze(self.critic_1(states_batch, actions_batch), 1)
            # 𝑄𝜃_2(𝑠,𝑎)
            critic_q2_on_batch = tf.squeeze(self.critic_2(states_batch, actions_batch), 1)

            # Q_target(sₜ,aₜ) =              r(sₜ,aₜ)       +     𝛾         V_soft(sₜ₊₁)
            critic_q_target = self.scale * rewards_batch + self.gamma * soft_target_values_next * (1 - terminals_batch)


            # 𝓛_critic_q1 = 1 / N (𝑄𝜃_1(𝑠,𝑎) - Q_target(sₜ,aₜ)) ^ 2
            critic_q1_loss = 0.5 * keras.losses.MSE(critic_q1_on_batch, critic_q_target)
            # 𝓛_critic_q2 = 1 / N (𝑄𝜃_2(𝑠,𝑎) - Q_target(sₜ,aₜ)) ^ 2
            critic_q2_loss = 0.5 * keras.losses.MSE(critic_q2_on_batch, critic_q_target)

        critic_1_network_gradient = tape.gradient(critic_q1_loss, self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_q2_loss, self.critic_2.trainable_variables)
        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, self.critic_2.trainable_variables))
        # ************************************************************************************************************


        # Soft-Update soft-target-value-Network with gradients of soft-value-network
        # Soft updates involve gradually updating the parameters of the target networks
        # towards the parameters of the online networks, rather than directly copying
        # the parameters of the online networks to the target networks.
        # Soft updates are used in SAC to stabilize the learning process by reducing the variance
        # in the target values and preventing the target networks from updating too quickly.
        # This can lead to more stable and efficient learning, especially in environments with
        # high-dimensional state and action spaces.
        self.soft_target_update()




    def append_transition(self, state, action, reward, new_state, terminal):
        self.replay_buffer.store_transition(state, action, reward, new_state, terminal)


    def hard_target_update(self):
        self._update_network_parameters(tau=1)

    def soft_target_update(self):
        self._update_network_parameters(tau=self.tau)

    def _update_network_parameters(self, tau):
        weights = []
        targets = self.soft_value_target_func.weights
        for i, weight in enumerate(self.soft_value_func.weights):
            weights.append(weight * tau + targets[i] * (1 - tau))

        self.soft_value_target_func.set_weights(weights)




    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.soft_value_func.save_weights(self.soft_value_func.checkpoint_file)
        self.soft_value_target_func.save_weights(self.soft_value_target_func.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.critic_1.load_weights(self.critic_1.checkpoint_file)
        self.critic_2.load_weights(self.critic_2.checkpoint_file)
        self.soft_value_func.load_weights(self.soft_value_func.checkpoint_file)
        self.soft_value_target_func.load_weights(self.soft_value_target_func.checkpoint_file)






