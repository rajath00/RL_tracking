# System import XXX
import random

import matplotlib.pyplot as plt

# External import
import numpy as np

# Datatype import
from matplotlib.axes import Axes

# Custom import
from replay_buffer import ReplayBuffer


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
        target_move=False,
    ):
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.target_move = False
        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.env = env
        self.replay_buffer = ReplayBuffer(1000, 2, self.action_size)
        self.batch_size = batch_size

        pass

    def training_loop(self):
        pass

    def train(self, num_episodes, max_steps, int_pos):

        if not self.target_move:
            pos = int_pos
            self.env.target_position(pos)

        for episode in range(num_episodes):

            print(episode)
            x, y = self.env.reset()
            done = False

            for step in range(max_steps):

                # exploration-exploitation tradeoff
                if random.uniform(0, 1) < self.epsilon:
                    # explore
                    action = self.env.action_space.sample()
                    print(f"random action {action}")
                else:
                    # exploit
                    q_values = self.model.forward((x, y))
                    action = np.argmax(q_values.detach().numpy())
                    print(f"q-action {action}")
                # take action and observe the reward
                new_state, reward, done, truncated = self.env.step(action)

                self.replay_buffer.store_transition(
                    (x, y), action, reward, new_state, done
                )
                x, y = new_state
                if done:
                    x, y = self.env.reset()
                    done = False

                if self.replay_buffer.mem_cntr < self.batch_size:
                    continue
                (
                    state,
                    action,
                    reward,
                    new_state,
                    done,
                ) = self.replay_buffer.sample_buffer(self.batch_size)

                # x = state[:,0]
                # y = state[:,1]
                # self.model.optimizer.zero_grad()

                mu = self.model.forward(state)

                # Q-learning algorithm

                target_value_ = self.target_model.forward(new_state)
                target = mu.clone()
                # target = np.zeros(np.shape(target_value_))
                # print(target_value_[0])ss
                for j in range(self.batch_size):

                    arg_max = np.argmax(target_value_[j].detach().numpy())
                    # print(mu[j])
                    # one_hot[j,arg_max] = 1
                    a = action[j]
                    target[j,a] = reward[j] + self.discount_rate * np.max(target_value_[j].detach().numpy()) * (1-done[j])
                    # print(reward[j])
                    # print(target[j])

                # one_hot[action] = 1
                # target = reward + (
                #     self.discount_rate * (np.max(self.model.forward(x_new, y_new)))
                # )

                # target = target * one_hot
                loss = self.loss_fn(mu, target)
                self.update_network_parameters()
                #
                # Update to our new state
                state = new_state
                # if done:
                #     break
            self.epsilon = np.exp(-self.decay_rate * episode)

    def update_network_parameters(self):
        # if tau is None:
        #     tau = self.tau

        model_params = self.model.named_parameters()
        # model_target_params = self.critic.named_parameters()

        model_state_dict = dict(model_params)
        # target_state_dict = dict(model_target_params)

        self.target_model.load_state_dict(model_state_dict)

    # def train_epoch(self,input):
    #     self.model.train()
    #     self.optimizer.zero_grad()
    #     z = self.model.forward(input).reshape(-1)
    #     loss = self.loss_fn(z, labels.float())
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     for episode in range(num_episodes):
