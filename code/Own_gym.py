# importing all the required libraries
import numpy as np
from gym import Env, spaces, utils
import random
from typing import Optional
from io import StringIO
from os import path
from gym.envs.toy_text.utils import categorical_sample

WINDOW_SIZE = (550, 350)


class Own_gym(Env):
    """
    Description
    # Boundary, the agent has to follow the Target while the target is also moving

     Map:

    +---------------------+
    |                     |
    |             T       |
    |                     |
    |    A                |
    |                     |
    +---------------------+


    Action Space
    0 - down
    1 - up
    2 - left
    3 - right



    Observation Space
    There are 50x50 discrete states where the Target may be.
    
    Each state space is represented by the tuple:
    (agent_row, agent_column, target_row, target_column)

    step`` and ``reset()`` will return an info dictionary that contains "p" and "action_mask" containing
    the probability that the state is taken and a mask of what actions will result in a change of state to speed up
    training.

    As Taxi's initial state is a stochastic, the "p" key represents the probability of the
    transition however this value is currently bugged being 1.0, this will be fixed soon.
    As the steps are deterministic, "p" represents the probability of the transition which is always 1.0

    For some cases, taking an action will have no effect on the state of the agent.
    In v0.25.0, ``info["action_mask"]`` contains a np.ndarray for each of the action specifying
    if the action will change the state.

    To sample a modifying action, use ``action = env.action_space.sample(info["action_mask"])``
    Or with a Q-value based algorithm ``action = np.argmax(q_values[obs, np.where(info["action_mask"] == 1)[0]])``.

    ### Rewards
    - -1 per step if distance between target and agent has increased
    - +20 for reaching target
    - +1  per step if the distance between target and agent has decreased

    ###Version History
    * v0: Initial version release
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(self, render_mode: Optional[str] = None, max_step=150):

        self.terminated = None
        self.num_states = 625
        self.num_rows = 25
        self.num_cols = 25
        max_row = self.num_rows - 1
        max_cols = self.num_cols - 1
        updated_target_pos = 30
        # self.initial_state_distribution = np.zeros(num_states)
        # print(self.initial_state_distribution)
        self.num_actions = 4
        self.P = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }
        self.action_call()
        # self.initial_state_distribution /= self.initial_state_distribution.sum()
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.s = None
        self.lastaction = None

    def action_call(self):
        for row in range(self.num_states):
            for action in range(self.num_actions):
                # function for getting rewards from target position
                    reward = -1
                    terminated = False
                    if action == 0:
                        if (row >= 0 and row <= 24):
                            reward = -100
                            new_state = row
                        else:
                            new_state = row - 25
                    elif action == 1:
                        if row >= 600 and row <= 624:
                            reward = -100
                            new_state = row
                        else:
                            new_state = row + 25
                    elif action == 2:
                        if row % 25 == 0:
                            reward = -100
                            new_state = row
                        else:
                            new_state = row - 1
                    else:
                        if row % 25 == 24:
                            reward = -100
                            new_state = row
                        else:
                            new_state = row + 1
                    self.P[row][action].append((new_state, reward, terminated))

    def target_position(self, target_pos):
        reward = 100
        terminated = True

        transitions = self.P[target_pos+25][0]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        s, r, t = transitions[i]

        self.P[target_pos + 25][0].append((s,reward,terminated))

        transitions = self.P[target_pos-25][1]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        s, r, t = transitions[i]

        self.P[target_pos - 25][1].append((s,reward,terminated))

        transitions = self.P[target_pos+ 1][2]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        s, r, t = transitions[i]

        self.P[target_pos + 1][2].append((s,reward,terminated))

        transitions = self.P[target_pos- 1][3]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        s, r, t = transitions[i]

        self.P[target_pos - 1][3].append((s,reward,terminated))

    def encode(self):

        pass

    def decode(self):

        pass

    def action_mask(self):
        pass

    def step(self, a):
        transitions = self.P[self.s][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        s, r, t = transitions[i]
        self.s = s
        self.lastaction = a

        return int(s), r, t, False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        self.s = np.random.randint(0, 625)
        self.lastaction = None

        return int(self.s)
