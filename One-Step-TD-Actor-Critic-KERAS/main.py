"""
Name : main.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 14.03.2023 16:39
Desc:
"""

import gym
from agent import Agent
import numpy as np
import matplotlib.pyplot as plt



# Entropy for Exploration is missing in this exmample.


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    agent = Agent(alpha=0.00001, beta=0.00005)

    env = gym.make('LunarLander-v2', render_mode="human")
    score_history = []
    num_episodes = 2000

    # T_transitions-Episodes
    for i in range(num_episodes):
        done = False
        score = 0
        # Sample initial State for episode
        state = env.reset()[0]

        # while state is not terminal
        while not done:

            # Sample action for state according to policy
            action = agent.choose_action(state)

            # Apply action to environment
            # Env gives me reward for action and samples the new state
            next_state, reward, done, info, _ = env.step(action)

            # Use datapoint for trainstep
            agent.train_step(state, reward, next_state, action, done)

            state = next_state
            score += reward

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print('episode ', i, 'score %.2f | average score %.2f' % (score, avg_score))

    filename = 'lunar-lander-actor-critic.png'


    x = [i+1 for i in range(num_episodes)]
    plot_learning_curve(x, score_history, filename)
