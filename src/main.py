# System import XXX
import torch.nn as nn

# External import

from torch import optim

from NeuralNetwork import NeuralNetwork

# Custom import
from Own_gym import Own_gym
from Q_Learning import Q_Learning
from Train import Train
from Test import Test

# Datatype import

LEARNING_RATE = 0.9
DISCOUNT_RATE = 0.8
EPSILON = 1
DECAY_RATE = 0.5
TARGET_MOVE = False
NO_ROWS = 10
NO_COLS = 10
INIT_POS = (5, 5)  # define initial position of the target
BATCH_SIZE = 10

Q_learning = False
Deep_learning = True


# create Taxi environment
env = Own_gym(NO_ROWS, NO_COLS)
num_episodes = 50
max_steps = 500  # per episode

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
    optimizer = optim.Adam(model.parameters(), lr=0.001)
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
        TARGET_MOVE,
    )
    train.train(num_episodes, max_steps, INIT_POS)

    test = Test(model, env)
    test.test(INIT_POS)

env.close()
