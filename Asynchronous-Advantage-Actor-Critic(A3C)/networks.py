"""
Name : networks.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 16.03.2023 14:06
Desc:
"""


# Check out https://github.com/MorvanZhou/pytorch-A3C

import torch as T
import torch.nn as nn
import  torch.nn.functional as F


# Global Optimizer. For loss calculation. Global Optimizer keeps track of updated gradients in each Thread
# Global Optimizer hands out newly updated parameters to all Agents in Threads, so they update always on the newest
# Parameters
# Sharding a global Optimizer for all our Agents.
class SharedAdam(T.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
            weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
                weight_decay=weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = T.zeros_like(p.data)
                state['exp_avg_sq'] = T.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dims, n_actions, gamma=0.99):
        super(ActorCriticNetwork, self).__init__()

        self.discount_factor = gamma

        # ACTOR-CRITIC NETWORK
        # ------------------------------------------------------------------------------#
        # We make seperate network for Actor and Critic, because we didn't
        # get a SharedNetwork workking with threading

        # Hidden layer of policy[Actor] DNN
        self.policy_hidden_lay = nn.Linear(*state_dims, out_features= 128)
        # Given a state, it outputs the probability distribution
        # over the n-output actions (n_actions)
        self.policy = nn.Linear(in_features=128, out_features=n_actions)

        # Hidden layer of value-function[Critic] DNN
        self.value_function_hidden_lay = nn.Linear(*state_dims, out_features=128)
        # This layer is the state-value-function approximation
        # It outputs the estimated state-value, for a given state
        self.value_function = nn.Linear(in_features=128, out_features=1)
        # ------------------------------------------------------------------------------#

        # In the standard A3C algorithm, the value function is estimated using a one-step lookahead,
        # meaning that it is estimated as the immediate reward plus the estimated value of the next state.
        # However, in an n-step version of A3C, we save our transitions in form of a N-step big Batch, with  regard
        # to, we do our update.
        self.rewards = []
        self.actions = []
        self.states = []

    # Append a new transition-tuple to our batch
    def append_experience_batch(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)


    # Resets our batch after update step.
    def clear_experience_batch(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self, state):
        # [ACTOR]
        # The state is passed to the first hidden layer of our Actor
        out_hidden_policy = F.relu(self.policy_hidden_lay(state))
        # Get our estimated distribution of actions for the state
        policy = self.policy(out_hidden_policy)
        # softmax needed to normalize distribution from 0.0 to 1.0
        policy = T.softmax(policy, dim=1)

        # [CRITIC]
        # The state is passed to the first hidden layer of our Critic
        out_hidden_state_val = F.relu(self.value_function_hidden_lay(state))
        # Get our estimated state value for the state
        state_value = self.value_function(out_hidden_state_val)

        return policy, state_value

    # calulate Returns from sequence of n-steps
    # calcNStepTargetForEachState
    # See https://subscription.packtpub.com/book/big-data-and-business-intelligence/9781788836579/8/ch08lvl1sec47/the-deep-n-step-advantage-actor-critic-algorithm
    def accumulate_N_step_targets_respectively(self, done):
        """
        Calculates the n-step return / target value for each state in the input-
        trajectory/n_step_transitions
        :param done: True if final state is a terminal state, if not False
        :return: The n-step return / target value for each state in the n_step_transitions
        """

        # All value have size of tensor [1,n_step]

        # accumulated states in our n-step-trajectory
        states = T.tensor(self.states, dtype=T.float)
        _, state_values = self.forward(states)

        # V(s') when not terminal state and 0, when terminal state
        # starting from the last step
        final_state_value = state_values[-1] * (1 - int(done))
        # Start from last step
        G = final_state_value

        n_step_return_batch = []
        # Start from the last step in the trajectory, rewersed rewards
        for reward in self.rewards[::-1]:
            # First Time is G(s_5) = r_5 + gamma*V(s_6)
            G = reward + self.discount_factor * G
            n_step_return_batch.append(G)

            # Next Time is G(s_4) = r_4 + gamma * G(s_5)
            # ....

        # From [G(s_5), G(s_4), G(s_3),...] -> [G(s_1), G(s_1), G(s_3),...]
        n_step_return_batch.reverse()
        n_step_return_batch = T.tensor(n_step_return_batch, dtype=T.float)

        # Each Value in n_step_return_batch is the target value for each state to minimise loss to
        return n_step_return_batch


    def accumulate_n_step_loss(self, done):
        # All of these values have a Tensor shape of [1, N-size]



        # accumulated states in our n-step-trajectory
        states = T.tensor(self.states, dtype=T.float)
        # accumulated action in our n-step-trajectory
        actions = T.tensor(self.actions, dtype=T.float)

        n_step_target_values = self.accumulate_N_step_targets_respectively(done)

        # accumulate policies and state_values for out n-step-trajectory
        policies, state_values = self.forward(states)
        state_values = state_values.squeeze()

        # adjust N-step-state-value /advantage-predicitions according to prediction error TD_Error
        # Basically the mean squarred error between the n-step-target_values and our state_values
        n_step_temporal_diff = n_step_target_values - state_values
        n_step_critic_loss = n_step_temporal_diff**2

        # Approximate Advantage with Temporal difference
        n_step_advantage = n_step_temporal_diff


        # Probability of taking actions
        dist = T.distributions.Categorical(policies)
        n_step_log_props = dist.log_prob(actions)

        # adjust probability of n-step-policy according to n-step advantage
        # Equivalent to log pi(a|s) * A(s,a), where A(s,a) (advantage) is approximated by Temporal Diffence
        n_step_actor_loss = -n_step_log_props*n_step_advantage

        # Take mean to normalize loss by the number of samples in a batch
        n_step_total_loss =  (n_step_critic_loss + n_step_actor_loss).mean()

        return n_step_total_loss

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float)
        policy, _ = self.forward(state)

        # sample an action according to our policy distribution
        dist = T.distributions.Categorical(policy)
        # [0] for reducing Tensor dimension
        action = dist.sample().numpy()[0]

        return action














