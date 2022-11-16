# System import XXX
import math
import random
# External import
import numpy as np
import torch

# Custom import
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
import os


class Train:
    def __init__(
        self,
        model,
        target_model,
        optimizer,
        loss_fn,
        env,
        learning_rate,
        discount_rate,
        epsilon,
        decay_rate,
        batch_size,
        buffer_size,
        target_move=False,
    ):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.target_move = target_move
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.env = env
        self.batch_size = batch_size
        self.buffer_size = buffer_size


    def train(self, num_episodes, max_steps, int_pos):

        if not self.target_move:
            pos = int_pos
            self.env.target_position(pos)
        self.model.train()
        eps4plot = []
        cumulative_ep_reward = []
        avg_ep_loss = []

        for episode in range(num_episodes):
            print(f"episode = {episode}")
            # print(episode)
            x, y = self.env.reset()
            done = False

            cnt = 1
            self.replay_buffer = ReplayBuffer(self.buffer_size, 2)
            ep_reward = 0
            ep_loss = 0
            num_losses_in_ep = 0
            self.target_model.load_state_dict(self.model.state_dict())
            for step in range(max_steps):

                # exploration-exploitation tradeoff
                if random.uniform(0, 1) < self.epsilon:
                    # explore
                    action = self.env.action_space.sample()

                else:
                    # exploit
                    q_values = self.model.forward((x, y))
                    action = np.argmax(q_values.detach().numpy())
                # take action and observe the reward
                new_state, reward, done, truncated = self.env.step(action)
                ep_reward += reward
                self.replay_buffer.store_transition(
                    (x, y), action, reward, new_state, done
                )
                x, y = new_state
                if done:
                    x, y = self.env.reset()
                    done = False
                state = new_state
                if self.replay_buffer.mem_cntr < self.buffer_size:   # self.batch_size:
                    continue
                if cnt == 1:
                    print(state)
                    cnt = 2
                (
                    state,
                    action,
                    reward,
                    new_state,
                    done,
                ) = self.replay_buffer.sample_buffer(self.batch_size)

                mu = self.model.forward(state)

                # Q-learning algorithm

                target_value_ = self.target_model.forward(new_state)
                target = mu.clone()
                # target = np.zeros(np.shape(target_value_))
                # print(target_value_[0])ss
                action = action.astype(int)
                for j in range(self.batch_size):

                    # arg_max = np.argmax(target_value_[j].detach().numpy())
                    a = action[j]
                    target[j,a] = reward[j] + self.discount_rate * np.max(target_value_[j].detach().numpy()) * (1-done[j])

                loss = self.loss_fn(mu, target)
                ep_loss += float(loss)
                num_losses_in_ep += 1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if episode % 10 == 0:

                self.target_model.load_state_dict(self.model.state_dict())
                torch.save(self.model.state_dict(), os.getcwd() + "/model.pt")
                self.epsilon = self.epsilon*self.decay_rate
                #
                # Update to our new state
            eps4plot.append(self.epsilon)


            cumulative_ep_reward.append(ep_reward)
            avg_ep_loss.append(200 * ep_loss)
        plt.figure()
        plt.plot(range(len(eps4plot)), eps4plot)
        plt.show()
        return cumulative_ep_reward, avg_ep_loss
    # def update_network_parameters(self):
    #
    #     model_params = self.model.named_parameters()
    #     model_state_dict = dict(model_params)
    #
    #     self.target_model.load_state_dict(model_state_dict)
