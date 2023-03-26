"""
Name : main.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 23.03.2023 12:23
Desc:
"""

import gym
import numpy as np
from agent import Agent

import matplotlib.pyplot as plt


# Implementation based on https://www.youtube.com/watch?v=hlv79rcHws0&t=2902
# Good implementation to compare to is https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/rl/ppo/experiment.py
# Multi Agent Version is supported by https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # Each agents wanders the environment for T_transitions steps and syncs  the transitions with
    # RolloutBatchBufferMaybeFedbyMultipleAgents Buffer, that has the size Transition-Steps T" * "T_transitions parallel Agents/Envs"
    # In our case the Buffer holds 20 transitions at a time.
    T_transitions = 20

    # After all agents have stepped their environment, the accumulated transitions are broken into
    # (T_transitions / minibatch_size) X minibatches
    minibatch_size = 5

    # We iterate over the all minibatches for n_epochs, updating our Actor-Critic-Network after each minibatch
    # In the form of n-step loss like A3C
    n_epochs = 4
    alpha = 0.0003

    # Can also use multiple parallel Actors -> A2C/A3C frameworks
    # We use only on Agent collecting transitions for T_transitions
    agent = Agent(n_actions=env.action_space.n, minibatch_size=minibatch_size,
                    alpha=alpha, epochs=n_epochs,
                    state_dims=env.observation_space.shape)
    MAX_N_GAMES = 300

    figure_file = 'plots/cartpole.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    # we play for MAX_N_GAMES
    for i in range(MAX_N_GAMES):
        # Sample initial State for episode
        state = env.reset()[0]
        done = False
        score = 0

        # While state is not terminal
        while not done:
            # Our Algorithm alternates between sampling transitions trough Agent(-s) and
            # updating Actor-Critic-Net for multiple epochs

            # Make a  trainition in our environment
            action, log_prob, state_value = agent.choose_action(state)
            next_state, reward, done, info, _ = env.step(action)
            n_steps += 1 # Increase step count
            score += reward

            # Save transition to our RolloutBuffer of size "T" * "Number Agents/Envs"
            agent.append_transition(state, action, log_prob, state_value, reward, done)

            # After environment has been stepped for "T_transitions", update Actor-Critic Network
            # on minibatches of our Rollout-Buffer for multiple epochs
            if n_steps % T_transitions == 0:
                agent.train_on_transitions()
                learn_iters += 1

            # next state in transition collections
            state = next_state

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
