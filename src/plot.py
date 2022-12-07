import numpy as np
from shapely.geometry import (
    Polygon,
    JOIN_STYLE,
)


def obstacles(ax, obs):
    polygon(ax, obs.polygon)


def boundary(ax, bound):
    polygon(ax, bound.polygon)


def polygon(ax, p):
    c = np.asarray(p.exterior.coords, dtype=np.float32)[:-1]
    c = np.vstack([c, c[0]])
    line(ax, c)


def line(ax, l):
    l = np.asarray(l)
    ax.plot(l[:, 0], l[:, 1])
