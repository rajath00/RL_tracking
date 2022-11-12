# System import XXX
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
# External import

from torch import optim


from NeuralNetwork import NeuralNetwork

# Custom import
from Own_gym import Own_gym
from Q_Learning import Q_Learning
from Train import Train
from Test import Test
from plot import display

# Datatype import

LEARNING_RATE = 0.9
DISCOUNT_RATE = 0.8
EPSILON = 1
DECAY_RATE = 0.99
TARGET_MOVE = False
NO_ROWS = 25
NO_COLS = 25
INIT_POS = (10,10)  # define initial position of the target
BATCH_SIZE = 20
BUFFER_SIZE = 500

Q_learning = False
Deep_learning = True

# create Taxi environment
env = Own_gym(NO_ROWS, NO_COLS)
num_episodes = 5000
max_steps = 520  # per episode

if Q_learning:

    learn = Q_Learning(
        env, LEARNING_RATE, DISCOUNT_RATE, EPSILON, DECAY_RATE, TARGET_MOVE
    )
    learn.train(num_episodes, max_steps, INIT_POS)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    learn.test(max_steps, INIT_POS)

    print(f"Test completed")
    input("Press Enter to view the Q-table...")
    learn.display()

if Deep_learning:

    model = NeuralNetwork()
    target_model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    train = Train(
        model,
        target_model,
        optimizer,
        loss_fn,
        env,
        LEARNING_RATE,
        DISCOUNT_RATE,
        EPSILON,
        DECAY_RATE,
        BATCH_SIZE,
        BUFFER_SIZE,
        TARGET_MOVE,
    )
    cum_ep_reward, avg_ep_loss = train.train(num_episodes, max_steps, INIT_POS)

    plt.figure()
    plt.plot(range(len(cum_ep_reward)), cum_ep_reward, label="Cumulative reward")

    qtable = np.zeros((NO_COLS,NO_ROWS,4))
    out = torch.zeros((4,1))
    for i in range(NO_COLS):
        for j in range(NO_ROWS):
            out=model.forward((i,j))
            qtable[i, j, :] = out.detach().numpy()

    display(qtable,NO_ROWS,NO_COLS)

    test = Test(model, env)
    test.test(INIT_POS)

env.close()
