"""
Name : agent.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 24.03.2023 17:05
Desc:
"""
import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from typing import Dict

from networks import ActorNetwork, CriticNetwork, RolloutBatchBufferMaybeFedbyMultipleAgents


class Agent(object):
    def __init__(self, n_actions, minibatch_size, state_dims, epochs, gamma=0.99, alpha=3e-4, epsilon_clip=0.2, gae_lambda = 0.95):
        self.discount_factor = gamma
        self.epsilon_clip = epsilon_clip
        self.epochs = epochs
        self.gae_lambda = gae_lambda
        self.c1 = 0.5


        # ACtor-Critic NETWORK
        # ------------------------------------------------------------------------------#
        # Dont use shared parameters for Actor/Critic, as it complicates loss function
        # for examle shown by "Machine Leanring with Phil".

        self.actor = ActorNetwork(n_actions=n_actions, state_dims=state_dims, alpha=alpha)
        self.critic = CriticNetwork(state_dims=state_dims, alpha=alpha)
        self.rollout_buffer = RolloutBatchBufferMaybeFedbyMultipleAgents(minibatch_size=minibatch_size)

    def append_transition(self, state, action, probs, vals, reward, done):
        self.rollout_buffer.append_transition(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)

        policy = self.actor(state)
        dist = Categorical(policy)
        action = dist.sample()

        log_prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()

        state_value = self.critic(state)
        state_value = T.squeeze(state_value).item()

        return action, log_prob, state_value


    def gae_advantage(self, rewards, state_values, terminals):
        """
        Calculates the Generalized n-step- advantage for each state state in
        the trajectory, as a weighted sum of all possible N-Step Advantages (A^1 + λ*A^2 + λ*A^3 + .... ) per Advantage-Time-Step.

        :param rewards: rewards encountered during rollouts
        :param state_values: state_values encountered during rolllouts
        :param terminals: final state bools encountered during rollouts
        :return: GAE for each state in rollout -> size 20
        """

        # length of our n-step rollout
        n_steps = len(rewards)

        # GAE for each step in rollout
        advantages = np.zeros(n_steps, dtype=np.float32)

        # As we begin from reversed, and A_t(last) = delta_t + gamma*lambda* A_t(next)(=0)
        last_advantage = 0

        # starting from the last step
        last_value = state_values[-1]
        for t in reversed(range(n_steps)):

            mask = 1.0 - terminals[t]

            # V(s+1) when not terminal state and 0, when terminal state
            last_value = last_value * mask

            # A(t+1) when not terminal state and 0, when terminal state
            last_advantage = last_advantage * mask

            #δ_t = r_t + γ * V(s+1) - V(s)
            delta = rewards[t] + self.discount_factor * last_value - state_values[t]

            #A_t = δ_t + γλ*A(t+1)
            # We roll from reversed -> smart
            last_advantage = delta + self.discount_factor * self.gae_lambda * last_advantage


            advantages[t] = last_advantage

            last_value = state_values[t]

        return advantages

    # After Data collection of all Agents has finished, update Actor-Critic network on minibatches of the
    # accumulated Rollout-Buffer of size "T x N_Agents " for multiple epochs
    def train_on_transitions(self):
        """

            After Data collection of all Agents has finished, update Actor-Critic network on minibatches of the
            accumulated Rollout-Buffer of size "T x N_Agents " for multiple epochs

        :return:
        """

        # get Accumulated RolloutBuffer, in our case of size T_transitions=20
        rollouts = self.rollout_buffer.rollouts()

        # As the paper suggests, we can calculate the advantage during transition collection (save it ) or complete before the update step.
        # Instead of using the N-Step Advantage of A3C, we use the Generalized Advantage Estimate, which recovers the sum of all possible
        # N-Step Advantages per Advantage-Time-Step.
        advantages = T.tensor(self.gae_advantage(rollouts["rewards"], rollouts["state_values"], rollouts["terminals"])).to(self.actor.device)

        # data to gpu
        state_values = T.tensor(rollouts["state_values"]).to(self.actor.device)


        # Update Actor & Critic over K epochs with minibatch gradient descent

        # After all agents have stepped their environment, the accumulated transitions are broken into
        # (T_transitions / minibatch_size) X minibatches
        # We iterate over the all minibatches for n_epochs, updating our Actor-Critic-Network after each minibatch
        # In the form of n-step loss like A3C


        # Our accumulated rollouts are broken into (T_transitions / minibatch_size) number of minibatches
        # We iterate over these minibatches, updating our Actor-Critic-Network in each minibatch in a multi step update
        # we know from A3C for multiple epochs. We are allowed to multi iterate the same data, because we constrain possible
        # bad policy updates, and therefore increase sample efficiency
        for _ in range(self.epochs):

            # Breaks our rollouts into (T_transitions / minibatch_size) number of minibatches of size minibach_size.
            # Ensures that all samples are used once during an epsiode
            # Samples are randomly shuffled, so that we minimize correlation between samples and maximize learning effect
            for minibatch_index in self.rollout_buffer.generate_indices_of_minibatches_from_rollouts():
                minibatch = {}
                minibatch["states"] = T.tensor(rollouts["states"][minibatch_index], dtype=T.float).to(self.actor.device)
                minibatch["log_probs_old"] = T.tensor(rollouts["log_probs"][minibatch_index]).to(self.actor.device)
                minibatch["actions"] = T.tensor(rollouts["actions"][minibatch_index]).to(self.actor.device)
                minibatch["advantages"] = advantages[minibatch_index]

                # As A ≈ V_target - V -> V_target = A + V
                minibatch["targets_returns"] = advantages[minibatch_index] + state_values[minibatch_index]

                self.train_minibatch(minibatch)

        # Clear rollouts, as we switch to the phase of data collection again.
        self.rollout_buffer.clear_all()




    def train_minibatch(self, minibatch: Dict[str, T.tensor]):
        """
            We update our Actor-Critic-Network with a gradient descent according to a minibatch.
            All calulation are with respect to each transition-tuples in the accumulated trajectories of a minibatch,
            therefore all values here have the Tensor-size of 'minibatch_size'.

            We use the difference between log of the probablities, not the division of the probablities like in the paper.
            Both things are equivalent, but logs tend to be more used more often because of numerically stability.

        :param states_samples:
        :param advantages_samples:
        :param log_policy_old_samples:
        :param actions_samples:
        :param returns_samples:
        :return:
        """

        # accumulate estimated policies and state values for out n-step-trajectory,
        # both are based on our current during this minibatch evolving new policy
        policies = self.actor(minibatch["states"])
        cur_state_values = T.squeeze(self.critic(minibatch["states"]))

        # New Data is only introduced through log_policy_new as the policy is updated.
        # log_policy_old is based on on the trajectories collected,
        # as we do multiple update steps on the same data essentially and constrain it with regard to the OLD policy.
        log_policy_new = Categorical(policies).log_prob(minibatch["actions"])
        log_policy_old = minibatch["log_probs_old"]



        # Probabilty ratio, which estimates the divergence between the old policy according to trajectory and the new one.
        # When we start running the optimization, the new policy is the same as the old policy, which means the ratio is 1.
        #  So at first, none of our updates will be clipped and we are guaranteed to learn something from these examples.
        #  However, as we update π using multiple epochs, the objective will start hitting the clipping limits, the gradient will
        #  go to 0 for those samples, and the training will gradually stop... until we move on to the next iteration and collect new samples.
        # So basically after the first update its Off-Policy (as we use outdated policy parameters, for targets")
        prob_ratio = T.exp(log_policy_new - log_policy_old)

        # r_t(θ) * A
        surrogate = prob_ratio * minibatch["advantages"]

        # clip(r_t(θ), 1-ε, 1+ε) * A
        clipped_surrogate = T.clamp(prob_ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * minibatch["advantages"]

        # We use mean, to turn our n-step-loss into a constant, averaged over all states.
        # The minus is needed, as we essentially want to do gradient ASCENT on it
        actor_loss = -T.min(surrogate, clipped_surrogate).mean()

        # We use mean, to turn our n-step-loss into a constant, averaged over all states.
        # V_targets are sampled from minibatches. As we update multiple epochs, we want the
        # current estimation of the state_value closer to our targets.
        # see https://math.stackexchange.com/questions/4328736/critic-loss-in-ppo
        critic_loss = ((minibatch["targets_returns"] - cur_state_values)**2).mean()

        # Entropy-term c_2*S is missing, as Phil never uses it. He argues we get naturally exploration anyways,
        # as we use a stochastic policy.
        total_loss = actor_loss + self.c1 * critic_loss


        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()





















