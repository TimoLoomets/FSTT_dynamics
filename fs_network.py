import tensorflow as tf
import numpy as np
from tqdm import tqdm

import random
import time
from os import path, makedirs

from fsenv import FSEnv
from dqn_agent import DQNAgent
from constants import *

epsilon = 0.5

if __name__ == "__main__":
    start_time = time.time()
    env = FSEnv()

    ep_rewards = [-200]

    # For more repetitive results
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)

    if not path.isdir('models'):
        makedirs('models')

    agent = DQNAgent()

    for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
        # Update tensorboard step every episode
        agent.tensorboard.step = episode

        # Restarting episode - reset episode reward and step number
        episode_reward = 0
        step = 1

        # Reset environment and get initial state
        current_state = env.reset()

        # Reset flag and start iterating until episode ends
        done = False
        while not done:
            if np.random.random() > epsilon:
                # Get action from Q table
                qualities = agent.get_qs(current_state)
                # qualities = action_qualities[:, 2]
                best_index = np.argmax(qualities)
                action = ACTIONS[best_index]  # [:2]
            else:
                # Get random action
                action = np.array([random.random() * 2 - 1,
                                   random.random() * 2 - 1])  # np.random.randint(0, env.ACTION_SPACE_SIZE)  # np.array(act, dtype=dt)

            new_state, reward, done = env.step(action)

            # Transform new continous state to new discrete state and count reward
            episode_reward += reward

            # if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            #    env.render()

            agent.update_replay_memory((current_state, action, reward, new_state, done))
            agent.train(done, step)
            current_state = new_state
            step += 1

        ep_rewards.append(episode_reward)
        if not episode % AGGREGATE_STATS_EVERY or episode == 1:
            average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(ep_rewards[-AGGREGATE_STATS_EVERY:])
            min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
            max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
            agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward,
                                           epsilon=epsilon)

            # Save model, but only when min reward is greater or equal a set value
            print("rewards: ", min_reward, average_reward, max_reward)
            if average_reward >= MIN_REWARD:
                agent.model.save(
                    f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

        # Decay epsilon
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

    end_time = time.time()
    print("total time:", end_time - start_time)
