"""
Name : agent.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 17.03.2023 15:05
Desc:
"""

import gym
import torch as T
from networks import ActorCriticNetwork



# Agent derives from multiprocess class, so i can run in its own thread
class Agent(T.multiprocessing.Process):
    def __init__(self, global_actor_critic, shared_optimizer, state_dims, n_actions, gamma,
                 thread_index, global_games_played, env_name, STEP_MAX_UPDATE, MAX_N_GAMES):
        super(Agent, self).__init__()
        self.local_ActorCriticNetwork = ActorCriticNetwork(state_dims=state_dims,
                                                           n_actions=n_actions,
                                                           gamma=gamma)

        self.globally_shared_ActorCritiNetwork = global_actor_critic
        self.agent_name = 'Agent%02i' % thread_index
        self.global_games_played  = global_games_played
        self.local_env = gym.make(env_name)
        self.globally_shared_optimizer = shared_optimizer
        self.MAX_N_GAMES = MAX_N_GAMES
        self.STEP_MAX_UPDATE = STEP_MAX_UPDATE

    def run(self) -> None:
        n_step  = 1

        # All agents are only allowed to play MAX_N_GAMES
        # The counter is tracked globally across all agent instances
        while self.global_games_played.value < self.MAX_N_GAMES:
            done = False
            score = 0
            state = self.local_env.reset()[0]
            self.local_ActorCriticNetwork.clear_experience_batch()

            # While state is not terminal
            while not done:

                # Make a transition in our local copy of environment according to our local network.
                action = self.local_ActorCriticNetwork.choose_action(state)
                next_state, reward, done, info, _ = self.local_env.step(action)
                score += reward

                # save transition to our N-step big batch
                self.local_ActorCriticNetwork.append_experience_batch(state, action, reward)

                # After n-steps of raoming in our environment, use the accumulated n-step-batch-of-Transitions for the
                # Update of our Actor and Critic (instead of doing after every transition)
                # The updated model local model parameters are pushed asynchronous to the global network.
                if  n_step % self.STEP_MAX_UPDATE == 0 or done:

                    # Calculate the with respect to the accumulated batch of Transitions,
                    # with respect to each encounteded state
                    batch_loss = self.local_ActorCriticNetwork.accumulate_n_step_loss(done)

                    # Prepare for new gradient-update
                    self.globally_shared_optimizer.zero_grad()

                    # compute the gradients (backprop) of the accumulated loss function with respect to the local ActorCritic-Network
                    batch_loss.backward()

                    # Synchronize our newly calculated local batch-gradients with the the global ActorCritic-Network
                    for local_param, global_param in zip(self.local_ActorCriticNetwork.parameters(), self.globally_shared_ActorCritiNetwork.parameters()):
                        global_param._grad = local_param.grad

                    # Update global model parameters based on newly synchronized gradients
                    self.globally_shared_optimizer.step()

                    # Pull most up to date model parameters from the shared global ActorCritic-Network, as it could have been updated
                    # in the mean time from another Agent in another Thread
                    # Make this Agent learn from others
                    self.local_ActorCriticNetwork.load_state_dict(
                        self.globally_shared_ActorCritiNetwork.state_dict()
                    )

                    # Clear our accumulated batch of Transitions.
                    self.local_ActorCriticNetwork.clear_experience_batch()

                # Increase step count
                n_step += 1
                state = next_state

            # Increase global games counter.
            with self.global_games_played.get_lock():
                self.global_games_played.value += 1

            print(self.agent_name, 'episode ', self.global_games_played.value, 'reward %.1f' % score)















        pass



