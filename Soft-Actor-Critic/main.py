"""
Name : main.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 31.03.2023 14:53
Desc:
"""
import gym


from collections import UserDict

import gym
import gym.envs.registration

# Do this before importing pybullet_envs (adds an extra property env_specs as a property to the registry, so it looks like the <0.26 envspec version)
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry

import pybullet_envs
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent


#***************************************************************************************************************************************************************
# For a waaaaay better implementation of SAC V1 see https://github.com/pranz24/pytorch-soft-actor-critic/blob/SAC_V/sac.py
# It has the SAC V2 with the ommited state value function and automiatic temperature tuning: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py
#***************************************************************************************************************************************************************



def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':

    env = gym.make('InvertedPendulumBulletEnv-v0')
    agent = Agent(state_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])

    n_games = 250
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(render_mode='human')

    for i in range(n_games):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, info= env.step(action)
            score += reward
            agent.append_transition(state, action, reward, next_state, done)
            if not load_checkpoint:
                agent.learn()
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
