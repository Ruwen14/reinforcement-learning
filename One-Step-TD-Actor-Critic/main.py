# WRITE A SHORT SUMMARY OF WHAT TYPE OF ACTOR_CRITIC THIS IS and the steps.



import gym
import numpy as np
from agent import Agent
from networks import ActorCriticSharedNetwork
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)



if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    agent = Agent(n_actions=env.action_space.n, shared_learning_rate=1e-5)
    n_games = 1800

    filename = 'cartpole2.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    load_checkpoint = False

    if load_checkpoint:
        agent.ActorCriticNet.load_model()


    # N-Episodes
    for game in range(n_games):
        # Sample initial State for episode
        state = env.reset()[0]
        done = False
        score = 0
        # while state is not terminal
        while not done:
            # Sample action for state according to policy
            action = agent.choose_action(state)
            # Apply action to environment
            # Env gives me reward for action and samples the new state
            next_state, reward, done, info, _ = env.step(action)
            score+= reward
            if not load_checkpoint:
                # Use datapoint for trainstep
                agent.train_step(state, reward, next_state, done)

            # Repeat for the state we just transitioned to
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.ActorCriticNet.save_model()


        print('episode ', game, 'score %.1f' % score, 'avg_score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)


