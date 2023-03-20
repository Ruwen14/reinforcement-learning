"""
Name : main.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 17.03.2023 15:05
Desc:
"""


import gym
import torch as T
from networks import ActorCriticNetwork, SharedAdam
from agent import Agent


# Q: Can this algorithm be adapted for continuous action spaces? The cartopole-v0 example you're using has a discrete action space If I recall correctly.
# A: Absolutely. Change the output of the actor layer to be a mean and standard deviation, and feed those into a normal distribution instead of categorical.
    # I haven't been able to get good hyperparameters that work with toy environments, however.


# Implementation based on https://www.youtube.com/watch?v=OcIx_TBu90Q&t=1370s
# For inspiration see additional:
    # https://github.com/MorvanZhou/pytorch-A3C/blob/5ab27abee2c3ac3ca921ac393bfcbda4e0a91745/continuous_A3C.py#L70
    # https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f
    # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-8-asynchronous-actor-critic-agents-a3c-c88f72a5e9f2#.8dufgx6aw
    # https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/train.py

# For A2C see https://github.com/grantsrb/PyTorch-A2C/tree/master/a2c



if __name__ == '__main__':
    lr = 1e-4
    env_name = 'CartPole-v0'
    n_actions = 2
    state_dims = [4]
    N_GAMES = 3000
    UPDATE_AFTER_N_STEPS = 10

    # Global Actor-Critic Network. This should approach a Optimum after N-Games,
    # which are played by our parallel agents.
    global_actor_critic = ActorCriticNetwork(state_dims=state_dims,
                                             n_actions=n_actions)
    global_actor_critic.share_memory()

    # Optimize with regard to our global Network. Parameters need to be shared.
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr,
                        betas=(0.92, 0.999))

    # Global counter so all agents combined dont play more than N_GAMES
    global_games_played = T.multiprocessing.Value('i', 0)

    # A set of Agents / Workers, that run in parallel in their own thread.
    # Each Agent has:
        # Local Environment to perform actions in
        # Local Actor-Critic Network, to which gradients are computed
        # Reference to globaly shared AC-Network
    parallel_agents = [
        Agent(global_actor_critic=global_actor_critic,
              shared_optimizer=optim,
              state_dims=state_dims,
              n_actions=n_actions,
              gamma=0.99,
              thread_index = i,
              global_games_played=global_games_played,
              env_name=env_name,
              STEP_MAX_UPDATE=UPDATE_AFTER_N_STEPS,
              MAX_N_GAMES=N_GAMES)
       for i in range(T.multiprocessing.cpu_count())
    ]

    # run in parallel and join() when finished (run() ended)
    [a.start() for a in parallel_agents]
    [a.join() for a in parallel_agents]
    print("Finished")
    print("Run trained model")



    # Inference to test trained global model.
    env = gym.make(env_name, render_mode="human")
    score_history = []
    while(True):
        done = False
        score = 0
        state = env.reset()[0]

        while not done:
            env.render()
            action = global_actor_critic.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            score += reward

            state = next_state

        print('score %.1f' % score)









