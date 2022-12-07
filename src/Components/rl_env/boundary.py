from typing import List


class Boundary:
    def __init__(self, bounds):
        self.minimum = [bounds[0], bounds[1]]
        self.maximum = [bounds[2], bounds[3]]

    def status(self, state) -> bool:
        if state[0] in range(self.minimum[0], self.maximum[0]) and state[1] in range(
            self.minimum[1], self.maximum[1]
        ):
            return False
        else:
            return True

    def get_obs(self):
        return self.minimum, self.maximum
