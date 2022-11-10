# System import XXX
import random
# External import
import numpy as np
import matplotlib.pyplot as plt

# Custom import
from plot import display
# Datatype import
from matplotlib.axes import Axes


class Q_Learning:

    def __init__(self, env, learning_rate, discount_rate, epsilon, decay_rate,target_move = False):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.target_move = False
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.qtable = np.zeros((env.num_rows, env.num_cols, self.action_size))
        self.env = env

    def train(self, num_episodes, max_steps, int_pos):
        if not self.target_move:
            pos = int_pos
            self.env.target_position(pos)

        for episode in range(num_episodes):

            x, y = self.env.reset()
            done = False
            if self.target_move:
                x_target, y_target = int_pos
                self.env.target_position((x_target, y_target))
                x_turn, y_turn = 1, 1
            for step in range(max_steps):
                # if self.target_move:
                #     if step % 20 == 0:
                #         if y_target not in range(1, env.num_cols - 1):
                #             y_turn = -1 * y_turn
                #         elif x_target not in range(1, env.num_rows - 1):
                #             x_turn = -1 * x_turn
                #         y_target = y_target + turn_direction * 1
                #         # env.action_call()
                #         env.target_position((x_target, y_target))

                # exploration-exploitation tradeoff
                if random.uniform(0, 1) < self.epsilon:
                    # explore
                    action = self.env.action_space.sample()
                else:
                    # exploit
                    action = np.argmax(self.qtable[x, y, :])
                # take action and observe the reward
                new_state, reward, done, truncated = self.env.step(action)
                # print(new_state)

                x_new = new_state[0]
                y_new = new_state[1]
                # Q-learning algorithm
                self.qtable[x, y, action] = self.qtable[x, y, action] + self.learning_rate * (
                        reward + (self.discount_rate * np.max(self.qtable[x_new, y_new, :])) - self.qtable[x, y, action])
                #
                # Update to our new state
                state = new_state
                if done:
                    break
            self.epsilon = np.exp(-self.decay_rate * episode)

    def test(self, max_steps, int_pos):

        rewards = 0
        state = self.env.reset()
        x_target, y_target = int_pos
        turn_direction = 1

        for step in range(max_steps):

            if self.target_move:
                if step % 25 == 0:
                    if y_target in range(1, self.env.num_cols - 1):
                        y_target = y_target + turn_direction * 1
                    else:
                        x_target = x_target + 1
                        turn_direction = -1 * turn_direction
                    self.env.target_position((x_target, y_target))
                    print(f"{(x_target, y_target)} is the new target position")
            print(f"TRAINED AGENT")
            print("Step {}".format(step + 1))
            print(f"state{state}")
            x, y = state
            action = np.argmax(self.qtable[x, y, :])
            print(f"action = {action}")
            print(f"Q-value = {np.max(self.qtable[x, y, :])}")
            new_state, reward, done, truncated = self.env.step(action)
            rewards += reward
            # env.render()

            print(f"score: {rewards}")
            state = new_state

            if done:
                print(f"state{state}")
                break

    def display(self):

        display(self.qtable,self.env.num_rows,self.env.num_cols)


