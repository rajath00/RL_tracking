
import numpy as np


class Test:

    def __init__(self, model, env):
        self.model = model
        self.env = env

    def test(self, int_pos):
        x, y = self.env.reset()
        self.env.target_position(int_pos)

        done = False
        step = 1
        rewards = 0
        print("Test started")
        print(done)
        path = np.array([x,y])
        print(path)
        while not done:

            print(f"TRAINED AGENT")
            print("Step {}".format(step))
            print(f"state{x,y}")

            q_values = self.model.forward((x, y))
            action = np.argmax(q_values.detach().numpy())
            print(f"action = {action}")
            print(f"Q-value = {np.max(q_values.detach().numpy())}")
            new_state, reward, done, truncated = self.env.step(action,True)
            print(done)
            rewards += reward
            print(f"score: {rewards}")
            x, y = new_state
            new = np.array([x,y])
            path = np.vstack((path,new))
            step += 1
            if done or step>150:
                print(f"final state{x, y}")
                return path
