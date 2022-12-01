import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# https://ai.stackexchange.com/questions/35023/what-is-the-difference-between-a-loss-function-and-reward-penalty-in-deep-reinfo

# https://ai.stackexchange.com/questions/14167/can-supervised-learning-be-recast-as-reinforcement-learning-problem/14168#14168

# https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

# https://arxiv.org/pdf/1506.02438.pdf

# https://arxiv.org/pdf/1912.02875.pdf


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fcd1 = nn.Linear(2,10)
        self.fcd2 = nn.Linear(10,20)
        self.fcd3 = nn.Linear(20,10)
        self.fcd4 = nn.Linear(10,4)

    def forward(self,state):

        x = torch.tensor(state,dtype=torch.float64)
        x = self.fcd1(x.float())
        x = self.fcd2(F.relu(x))
        x = self.fcd3(F.relu(x))
        x = self.fcd4(F.relu(x))

        return x