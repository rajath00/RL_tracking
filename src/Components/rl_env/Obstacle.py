import numpy as np
from shapely.geometry import (
    Polygon,
    JOIN_STYLE,
)  # polygon = Polygon([(0, 0), (1, 1), (1, 0)])
from Components.rl_env.agent import Agent
from typing import Union
import matplotlib.pyplot as plt
import geopandas as gpd


def _exterior_nodes(polygon: Polygon):
    return np.asarray(polygon.exterior.coords, dtype=np.float32)[:-1]


class Obstacle:
    def __init__(self, extreme_points) -> None:
        self.corners = np.asarray(extreme_points, dtype=np.float32)
        self._padded_corners = None
        self.polygon = None
        self.create_polygon()

    def create_pad(self):
        polygon = Polygon(self.corners)
        self._padded_corners = _exterior_nodes(
            polygon.buffer(Agent.RADIUS, join_style=JOIN_STYLE.mitre, resolution=4)
        )
        return self._padded_corners

    def create_polygon(self):
        self.polygon = Polygon(self.create_pad())
        # self.polygon = Polygon(self.corners)

    def collision(self, agent) -> bool:
        return self.polygon.contains(agent.point)


class Boundary:
    def __init__(self, extreme_points) -> None:
        self.corners = np.asarray(extreme_points, dtype=np.float32)
        self.polygon = None
        self._padded_corners = None
        self.create_polygon()

    def create_polygon(self):
        self.polygon = Polygon(self.create_pad())
        # self.polygon = Polygon(self.corners)

    def collision(self, agent) -> bool:
        return not self.polygon.contains(agent.point)

    def create_pad(self):
        polygon = Polygon(self.corners)
        self._padded_corners = polygon.buffer(
            -Agent.RADIUS, join_style=JOIN_STYLE.mitre, resolution=4
        )
        return self._padded_corners


if __name__ == "__main__":

    agent = Agent((100, 100))
    obstacle = Obstacle([(0, 0), (100, 0), (100, 100), (0, 100)])
    boundary = Boundary([(-100, -100), (-100, 100), (100, 100), (100, -100)])
    # obstacle.create_polygon()
    print(obstacle.collision(agent))
    # boundary.create_polygon()
    print(boundary.collision(agent))

    print(_exterior_nodes(obstacle.polygon))

    # p = gpd.GeoSeries(
    #     obstacle.polygon,
    # )
    # p.plot()
    # q = gpd.GeoSeries(boundary.polygon)
    # q.plot()
    # plt.show()

    # plt.plot(_exterior_nodes(obstacle.polygon))
