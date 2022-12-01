import math

from typing import List
import random


class Target:
    def __init__(self, pos):

        self.x: int = pos[0]
        self.y: int = pos[1]
        # self.state: List[int,int] = [self.x, self.y]
        self.change_direction_x: int = 1
        self.change_direction_y: int = 1
        # self.bound_x = bounds(0)
        # self.bound_y = bounds(1)
        self.path_x = random.randint(-1, 1)
        self.path_y = random.randint(-1, 1)

    def step(self):

        self.x = self.x + self.path_x * 1
        self.y = self.y + self.path_y * 1
        # self.x = self.x + (1*self.change_direction_x)
        # if self.x not in range(self.bound_x):
        #     self.change_direction_x*=-1
        #     self.x = self.x + (1*self.change_direction_x)
        #     self.y = self.y + (1*self.change_direction_y)
        # if self.y not in range(self.bound_y):
        #     self.change_direction_y*=-1
        pass

    def get_obs(self):
        return self.x, self.y

    def calculate_dist(self, state) -> float:
        x = self.x - state[0]
        y = self.y - state[1]
        dist = math.sqrt(x**2 + y**2)
        return dist

    def reached_target(self, state) -> bool:
        if self.calculate_dist(state) < 1:
            return True
        else:
            return False
