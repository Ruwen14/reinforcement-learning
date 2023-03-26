"""
Name : inference.py
Author  : Ruwen Kohm
Contact : ruwen.kohm@web.de
Time    : 25.03.2023 14:23
Desc:
"""
import gym
import numpy as np
from agent import Agent


env = gym.make('CartPole-v0', render_mode="human")
score_history = []

N = 20
minibatch_size = 5
n_epochs = 4
alpha = 0.0003
agent = Agent(n_actions=env.action_space.n, minibatch_size=minibatch_size,
                    alpha=alpha, epochs=n_epochs,
                    state_dims=env.observation_space.shape)

agent.load_models()

while (True):
    done = False
    score = 0
    state = env.reset()[0]

    while not done:
        env.render()
        action, _, _2 = agent.choose_action(state)
        next_state, reward, done, info, _ = env.step(action)
        score += reward

        state = next_state

    print('score %.1f' % score)