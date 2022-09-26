import numpy as np
import gym
import random


def main():
    # create Taxi environment
    env = gym.make('Taxi-v3')

    # initialize q-table

    state_size = env.observation_space.n
    action_size = env.action_space.n

    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate = 0.005

    # training variables
    num_episodes = 500
    max_steps = 99 # per episode

    # training
    for episode in range(num_episodes):

        state = env.reset()
        done = False
        state = state[0]

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])

            # take action and observe the reward
            #print(env.step(action))
            new_state, reward, done, truncated, info = env.step(action)

            # Q-learning algorithm
            qtable[state, action] = qtable[state, action] + learning_rate * (
                        reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            # Update to our new state
            state = new_state

            if done:
                break

        epsilon = np.exp(-decay_rate * episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    env = gym.make('Taxi-v3', render_mode='human')
    state = env.reset()
    done = False
    rewards = 0
    state = state[0]
    for s in range(max_steps):

        print(f"TRAINED AGENT")
        print("Step {}".format(s + 1))

        action = np.argmax(qtable[state, :])

        new_state, reward, done, truncated, info = env.step(action)
        rewards += reward
        env.render()

        print(f"score: {rewards}")
        state = new_state

        if done:
            break

    env.close()


if __name__ == "__main__":
    main()