# importing all the required libraries\

# System import
from typing import Optional, Tuple

import gym

# External import
import numpy as np
from time import time

# Specific lib used
from gym import Env, spaces, utils
import math
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.patches as patches

import plot


class Own_gym(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    ax: Axes = None
    fig: Figure = None

    def __init__(self, generate_env):

        self.generate_env = generate_env

        self.terminated = None
        self.reached_goal = None  # total no of states
        self.num_actions = 4
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(
                    low=np.array(
                        [
                            -math.inf,
                            -math.inf,
                            -math.inf,
                            -math.inf,
                            -math.inf,
                            -math.inf,
                        ]
                    ),
                    high=np.array(
                        [math.inf, math.inf, math.inf, math.inf, math.inf, math.inf]
                    ),
                )
            }
        )
        self.time_step = 1
        self.run_time = 0

    # function to reset the position
    def reset(self, seed=None, options=None) -> dict:
        # super().reset(seed=seed)
        self.run_time = 0
        self.last_render_at = 0
        self.previous_distance = 0
        self.current_distance = 0

        self.agent, self.target, self.boundary, self.obstacle = self.generate_env()
        observation = self._get_obs()
        info = self._get_info()

        self.traversed_positions = [self.agent.position]

        return observation

    def _get_obs(self) -> dict:
        agent_obs = self.agent.get_obs()
        target_obs = self.target.get_obs()
        obs = {"obs": [*agent_obs, *target_obs]}
        return obs

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        self.run_time += 1
        self.previous_distance = self.target.calculate_dist(self.agent)
        self.agent.step(action, self.time_step)
        self.target.step()
        self._update_status()
        observation = self._get_obs()
        terminated = self.terminated or self.reached_goal
        reward = self._get_reward()

        self.traversed_positions.append(self.agent.position)

        info = self._get_info()
        return observation, reward, terminated, info

    def _get_reward(self) -> float:

        current_distance = self.target.calculate_dist(self.agent)
        # reward = 0
        reward = self.previous_distance - current_distance - (0.0010 * self.run_time)
        if self.terminated:
            reward = reward - 1000
        if self.reached_goal:
            reward = reward + 100
        return reward

    def _update_status(self):
        self.terminated = self.boundary.collision(
            self.agent
        ) or self.obstacle.collision(self.agent)
        self.reached_goal = self.target.reached_target(self.agent)

    def _get_info(self) -> dict:
        return {}

    def render(self, mode: str = "human") -> None:
        if mode != "human":
            return
        if self.fig is None or self.ax is None:
            if mode == "human":
                plt.ion()
            self.fig, self.ax = plt.subplots()

        self.ax.cla()

        dt = time() - self.last_render_at
        self.last_render_at = time()
        fps = 1 / dt

        plot.obstacles(self.ax, self.obstacle)
        plot.boundary(self.ax, self.boundary)
        plot.line(self.ax, self.traversed_positions)
        # minimum, maximum = self.boundary.get_obs()
        # rect1 = patches.Rectangle(
        #     (minimum[0], minimum[1]),
        #     maximum[0],
        #     maximum[1],
        #     color="blue",
        #     fc="none",
        #     lw=2,
        # )
        # self.ax.add_patch(rect1)
        # plt.xlim([minimum[0] - 20, maximum[0] + 20])
        # plt.ylim([minimum[1] - 20, maximum[1] + 20])

        self.ax.add_artist(plt.Circle((self.agent.position), 5, color="r", alpha=0.7))
        self.ax.add_artist(plt.Circle((self.target.get_obs()), 5, color="b", alpha=0.7))

        self.fig.canvas.draw()

        if mode == "human":
            self.fig.canvas.flush_events()
        # self.ax.cla()
        # self.fig.canvas.flush_events()


if __name__ == "__main__":

    # agent = Agent((100, 100))
    # obstacle = Obstacle([(250, 250), (250, 350), (350, 350), (350, 250)])
    # boundary = Boundary([(0, 0), (0, 500), (500, 500), (500, 0)])
    # target = Target([250, 250])
    pass
