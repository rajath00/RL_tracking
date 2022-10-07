import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.fcd1 = nn.Linear(2,10)
        self.fcd2 = nn.Linear(10,4)

    def forward(self,x):

        x = self.fcd1(x)
        x = self.fcd2(F.relu(x))
        x = F.softmax(F.relu(x))

        return x