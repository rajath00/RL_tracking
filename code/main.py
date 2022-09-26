import numpy as np
import gym
import random
from Own_gym import Own_gym
import matplotlib.pyplot as plt
import math
# create Taxi environment
env = Own_gym()

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
num_episodes = 50
max_steps = 500 # per episode
int_pos = 30
# training
for episode in range(num_episodes):

    state = env.reset()
    done = False
    pos = int_pos

    env.target_position(pos)

    for s in range(max_steps):
        if s % 25 == 0:
            pos = pos + 1
            #print(f"{pos} is the new target position")
            env.action_call()
            env.target_position(pos)
        # exploration-exploitation tradeoff
        if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state, :])
        # take action and observe the reward
        #print(env.step(action))
        new_state, reward, done, truncated = env.step(action)

        # Q-learning algorithm
        qtable[state, action] = qtable[state, action] + learning_rate * (
                    reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

        # Update to our new state
        state = new_state
        if done:
            break
    epsilon = np.exp(-decay_rate * episode)
V = []
for i in range(625):
    V.append(round(max(qtable[i])))

V_shape = np.reshape(V,(25,25))
print(np.shape(qtable))
print(V_shape)
plt.table(cellText=V_shape,rowLabels=None,colLabels=None)
plt.show()
print(f"Training completed over {num_episodes} episodes")
input("Press Enter to watch trained agent...")

env = Own_gym()
state = env.reset()
done = False
rewards = 0
pos = 30
env.target_position(pos)

for s in range(max_steps):
    if s % 25 == 0:
        pos = pos + 1
        print(f"{pos} is the new target position")
        env.action_call()
        env.target_position(pos)

    print(f"TRAINED AGENT")
    print("Step {}".format(s + 1))
    print(f"state{state}")

    action = np.argmax(qtable[state, :])

    new_state, reward, done, truncated = env.step(action)
    rewards += reward
    #env.render()

    print(f"score: {rewards}")
    state = new_state

    if done:
        break

env.close()