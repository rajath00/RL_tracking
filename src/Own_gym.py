# importing all the required libraries
# System import
from typing import Optional
# External import
import numpy as np
# Specific lib used
from gym import Env, spaces, utils
from gym.envs.toy_text.utils import categorical_sample

WINDOW_SIZE = (550, 350)

# XXX
'''
Important in OpenAI Gym: "gym.spaces" folder, "core.py"
How to generate P? Better idea? From occupancy map?
How to load hyper-parameters? Use yaml (better, more readable) or json (faster)
'''


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
    There are 25x25 discrete states where the Target may be.
    
    Each state space is represented by the tuple:
    (agent location between 0 and 625)

    step`` and ``reset()`` will return an info dictionary

    ### Rewards
    - -1 per step
    - +100 for reaching target
    - -100 for going out of bounds

    ###Version History
    * v0: Initial version release
    """

    # metadata = {
    #     "render_modes": ["human", "ansi", "rgb_array"],
    #     "render_fps": 4,
    # }

    def __init__(self, render_mode: Optional[str] = None, max_step=150):

        self.terminated = None
        self.num_states = 625   # total no of states
        self.num_rows = 25
        self.num_cols = 25
        max_row = self.num_rows - 1
        max_cols = self.num_cols - 1
        #updated_target_pos = 30
        self.num_actions = 4
        self.P = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }
        self.action_call()
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.state = None
        self.last_action = None

    # function for getting setting rewards for all the states
    def action_call(self):
        for row in range(self.num_states):
            for action in range(self.num_actions):
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

    # function to set the reward func around the target.
    def target_position(self, target_pos): # XXX what is target_pos? int or tuple? it should be tuple (x,y)
        reward = 50 # XXX why 50
        terminated = False

        for j in range(1,4):
            if j==1:
                terminated = True
                reward = 100
            transitions = self.P[target_pos + (25*j)][0]
            i = categorical_sample([t[0] for t in transitions], self.np_random)  # XXX this can be changed to Pytorch function later
            s, r, t = transitions[i]
            self.P[target_pos + (25*j)][0] = [(s, reward, terminated)]

            transitions = self.P[target_pos - (25*j)][1]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            s, r, t = transitions[i]

            self.P[target_pos - (25*j)][1] = [(s, reward, terminated)]

            transitions = self.P[target_pos + (1*j)][2]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            s, r, t = transitions[i]

            self.P[target_pos + (1*j)][2] = [(s, reward, terminated)]

            transitions = self.P[target_pos - (1*j)][3]
            i = categorical_sample([t[0] for t in transitions], self.np_random)
            s, r, t = transitions[i]

            self.P[target_pos - (1*j)][3] = [(s, reward, terminated)]

    def encode(self):

        pass

    def decode(self):

        pass

    def action_mask(self):
        pass

    # function to return the next state, reward and termination criteria
    def step(self, a):
        transitions = self.P[self.state][a]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        s, r, t = transitions[i]
        self.state = s
        self.last_action = a

        return int(s), r, t, False

    # function to reset the position
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        self.state = np.random.randint(0, 625)
        self.last_action = None

        return int(self.state)
