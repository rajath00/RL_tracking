import math

from typing import List
import random
from Components.rl_env.agent import Agent


class Target:
    def __init__(self, pos):

        self.x: int = pos[0]
        self.y: int = pos[1]
        self.change_direction_x: int = 1
        self.change_direction_y: int = 1
        self.path_x = random.randint(-1, 1)
        self.path_y = random.randint(-1, 1)

    def step(self):

        self.x = self.x + random.randint(-1, 1) * 2
        self.y = self.y + random.randint(-1, 1) * 2

    def get_obs(self):
        return self.x, self.y

    def calculate_dist(self, agent: Agent) -> float:
        x = self.x - agent.position[0]
        y = self.y - agent.position[1]
        dist = math.sqrt(x**2 + y**2)
        return dist

    def reached_target(self, agent: Agent) -> bool:
        if self.calculate_dist(agent) < 5:
            return True
        else:
            return False


if __name__ == "__main__":

    agent = Agent((50, 100))
    target = Target((50, 50))
    print(agent.position)
    g = target.calculate_dist(agent)
