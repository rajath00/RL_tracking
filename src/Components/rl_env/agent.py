from typing import List
from shapely.geometry import Point
from math import cos, sin
import numpy as np
import numpy.typing as npt


class Agent:

    RADIUS = 0.5
    MAX_VEL = 1.5
    MIN_VEL = -0.5
    MAX_ANG_VEL = 0.5
    MIN_ANG_VEL = -0.5
    # MAX_ACC = 1.0
    # MIN_ACC = -1.0
    # MAX_ANG_ACC = 3.0
    # MIN_ANG_ACC = -3.0

    _point: Point
    _position: npt.NDArray[np.float32]

    def __init__(self, position) -> None:
        self.position = np.asarray(position, dtype=np.float32)
        self.speed = 0.0
        self.angle = 0.0
        # self.angular_velocity = 0.0

    @property
    def position(self) -> npt.NDArray[np.float32]:
        return self._position

    @position.setter
    def position(self, position):
        self._position = position
        self._point = Point(self._position)

    @property
    def point(self) -> Point:
        return self._point

    def step(self, action: int, time_step: float) -> None:

        # Action 1: Turn Left ,
        # Action 2: Turn Right
        # Action 3: Keep same direction
        # Action 4: Move back
        pos = 0.0
        if action == 0:
            self.angle = time_step * Agent.MIN_ANG_VEL
            pos = time_step * Agent.MAX_VEL

        elif action == 1:
            self.angle = time_step * Agent.MAX_ANG_VEL
            pos = time_step * Agent.MAX_VEL

        elif action == 2:
            self.angle = time_step * 0
            pos = time_step * Agent.MAX_VEL

        elif action == 3:
            self.angle = time_step * 0
            pos = time_step * Agent.MIN_VEL

        self.position[0] += (pos) * cos(self.angle)
        self.position[1] += (pos) * sin(self.angle)

    def get_obs(self):
        return self.position[0], self.position[1], self.angle, self.speed
