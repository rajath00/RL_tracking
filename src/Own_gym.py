# importing all the required libraries\

# System import
from typing import Optional
# External import
import numpy as np
# Specific lib used
from gym import Env, spaces, utils



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
    * v2: Initial version release
    """

    # metadata = {
    #     "render_modes": ["human", "ansi", "rgb_array"],
    #     "render_fps": 4,
    # }

    def __init__(self,num_rows,num_cols):

        self.terminated = None
        self.num_states = 625  # total no of states
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.x_cor = 0
        self.y_cor = 0
        self.actions = {0, 1, 2, 3}
        self.num_actions = 4
        self.P = np.zeros((self.num_rows, self.num_cols, 2))
        self.target_pos = np.array([int(self.num_rows / 2), int(self.num_cols / 2)])
        self.action_call()
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_states)
        self.state = (None)
        self.last_action = None

    # function for getting setting rewards for all the states

    # def create_env(self):
    #
    #     x_cor = np.arange(0, self.num_rows)
    #     y_cor = np.arange(0, self.num_cols)
    #     self.x_cor, self.y_cor = np.meshgrid(x_cor, y_cor)

    def action_call(self):
        obstacles = np.ones((self.num_rows, self.num_cols), dtype=bool)
        obstacles[1:self.num_rows - 1, 1:self.num_cols - 1] = 0

        free = np.zeros((self.num_rows, self.num_cols), dtype=bool)
        free[1:self.num_rows - 1, 1:self.num_cols - 1] = 1

        self.P[:, :, 0][obstacles] = -100  # reward assignment
        self.P[:, :, 0][free] = -5 # reward assignment
        self.P[:, :, 1][obstacles] = 1  # termination
        self.P[:, :, 1][free] = 0  # termination

    # function to set the reward func around the target
    def target_position(self, new_pos):  # XXX what is target_pos? int or tuple? it should be tuple (x,y)
        reward = 50  # XXX why 50
        extend_mask = 3
        self.P[extend_mask + self.target_pos[0]:self.target_pos[0] - extend_mask, extend_mask + self.target_pos[1]:self.target_pos[1] - extend_mask, 0] = -1
        self.P[self.target_pos[0], self.target_pos[1], 1] = 0

        self.P[extend_mask + new_pos[0]:new_pos[0] - extend_mask, extend_mask + new_pos[1]:new_pos[1] - extend_mask, 0] = reward
        self.P[new_pos[0], new_pos[1], 0] = 100
        self.P[new_pos[0], new_pos[1], 1] = 1

        self.target_pos = new_pos

    def encode(self):

        pass

    def decode(self):

        pass

    def action_mask(self):
        pass

    # function to return the next state, reward and termination criteria
    def step(self, a):

        next_state = (0, 0)
        r = 0
        t = 0
        x = self.state[0]
        y = self.state[1]
        if a in self.actions:
            if a == 0:
                next_state = (x - 1, y)
            elif a == 1:
                next_state = (x + 1, y)
            elif a == 2:
                next_state = (x, y - 1)
            else:
                next_state = (x, y + 1)

        self.state = next_state
        x = self.state[0]
        y = self.state[1]
        self.last_action = a
        return self.state, self.P[x, y, 0], self.P[x, y, 1], False

    # function to reset the position
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        init_state = np.random.randint(1,self.num_rows-1, size=(1, 2))
        # print(init_state)
        self.state = init_state[0]
        self.last_action = None

        return self.state
