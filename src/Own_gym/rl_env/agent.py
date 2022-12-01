from typing import List


class Agent:

    def __init__(self, pos):

        self.x = pos[0]
        self.y = pos[1]
        # self.state = pos

    def step(self,action: int):

        if action == 0:
            self.x = self.x - 1
        elif action == 1:
            self.x = self.x + 1
        elif action == 2:
            self.y = self.y - 1
        else:
            self.y= self.y + 1

    def get_obs(self):
        return self.x, self.y
