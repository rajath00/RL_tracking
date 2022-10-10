# System import XXX

# External import
import torch.nn as nn
from torch import optim
# Custom import
from Own_gym import Own_gym
from NeuralNetwork import NeuralNetwork
from Q_Learning import Q_Learning
from Train import Train
# Datatype import

LEARNING_RATE = 0.9
DISCOUNT_RATE = 0.8
EPSILON = 1
DECAY_RATE = 0.5
TARGET_MOVE = False
NO_ROWS = 25
NO_COLS = 25
INIT_POS = (5,15) # define initial position of the target

Q_learning = False
Deep_learning = True


# create Taxi environment
env = Own_gym(NO_ROWS,NO_COLS)

if Q_learning:

    learn = Q_Learning(env,LEARNING_RATE,DISCOUNT_RATE,EPSILON,DECAY_RATE,TARGET_MOVE)
    num_episodes = 1000
    max_steps = 300   # per episode
    learn.train(num_episodes,max_steps,INIT_POS)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    learn.test(max_steps,INIT_POS)

    print(f"Test completed")
    input("Press Enter to view the Q-table...")
    learn.display()

if Deep_learning:

    model = NeuralNetwork()
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    Train(model,optimizer,)




env.close()