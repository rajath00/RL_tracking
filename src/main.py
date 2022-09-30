# System import XXX
import random
# External import
import numpy as np
import matplotlib.pyplot as plt
# Custom import
from Own_gym import Own_gym
# Datatype import
from matplotlib.axes import Axes


# Function to plot the heat map for the max Q-values in each state
def plot_values(ax: Axes, V: np.ndarray):
    # reshape the state-value function
    V = np.reshape(V,(25,25))
    V = np.flip(V)
    # plot the state-value function
    im = ax.imshow(V, cmap='cool')
    for (j,i),label in np.ndenumerate(V):
        ax.text(i, j, np.round(label,1), ha='center', va='center', fontsize=5)
    ax.tick_params(bottom='off', left='off', labelbottom='off', labelleft='off')

# create Taxi environment
env = Own_gym()

# initialize q-table
state_size = env.observation_space.n
action_size = env.action_space.n
qtable = np.zeros((state_size, action_size))

# hyper-parameters
learning_rate = 0.9
discount_rate = 0.8
epsilon = 1.0
decay_rate = 0.005
target_move = True

# training variables
num_episodes = 1000
max_steps = 150  # per episode
int_pos = 199  # define initial position of the target

if not target_move:
    pos = int_pos
    env.target_position(pos)
# training

for episode in range(num_episodes):

    state = env.reset()
    done = False
    if target_move:
        pos = int_pos
        env.target_position(pos)

    for step in range(max_steps):

        if target_move:
            if step % 20 == 0:
                pos = pos + 1
                env.action_call()
                env.target_position(pos)

        #exploration-exploitation tradeoff
        if random.uniform(0,1) < epsilon:
            # explore
            action = env.action_space.sample()
        else:
            # exploit
            action = np.argmax(qtable[state, :])

        # take action and observe the reward
        new_state, reward, done, truncated = env.step(action)

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

# testing
env = Own_gym()
state = env.reset()
done = False
rewards = 0

env.target_position(int_pos)
pos = int_pos
for step in range(max_steps):

    if target_move:
        if step % 25 == 0:
            pos = pos + 1
            print(f"{pos} is the new target position")
            env.action_call()
            env.target_position(pos)

    print(f"TRAINED AGENT")
    print("Step {}".format(step + 1))
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

#disply Final Q table
Q = [np.max(qtable[x]) for x in range(625)]
Q = np.reshape(Q,[625,1])
fig1, ax1 = plt.subplots(figsize=(10,3))
plot_values(ax1, Q)
plt.show()
