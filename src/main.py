# System import XXX
import gym
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env

# External import

# Custom import
from Components.rl_env.agent import Agent
from Components.rl_env.target import Target

# from Own_gym.rl_env.boundary import Boundary
import random
from Components.rl_env.Obstacle import Obstacle, Boundary

# Datatype import

# LEARNING_RATE = 0.9
# DISCOUNT_RATE = 0.5
# EPSILON = 1
# DECAY_RATE = 0.9
# TARGET_MOVE = False
# NO_ROWS = 50
# NO_COLS = 50
# INIT_POS = (35,35)  # define initial position of the target
# BATCH_SIZE = 50
# BUFFER_SIZE = 2000


# Q_learning = False
Deep_learning = True
#
# # create Taxi environment
# env = Own_gym(NO_ROWS, NO_COLS)
# num_episodes = 1000
# max_steps = 2100  # per episode

# if Q_learning:
#
#     learn = Q_Learning(
#         env, LEARNING_RATE, DISCOUNT_RATE, EPSILON, DECAY_RATE, TARGET_MOVE
#     )
#     learn.train(num_episodes, max_steps, INIT_POS)
#
#     print(f"Training completed over {num_episodes} episodes")
#     input("Press Enter to watch trained agent...")
#
#     learn.test(max_steps, INIT_POS)
#
#     print(f"Test completed")
#     input("Press Enter to view the Q-table...")
#     learn.display()

if __name__ == "__main__":

    def env_components():

        # g = random.randint(-240, 240)
        # agent = Agent([250 + g, 250 + g])
        # h = random.randint(-100, 100)
        # target = Target([250 + h, 250 + h])
        # boundary = Boundary([0, 0, 500, 500])
        agent = Agent((100, 100))
        obstacle = Obstacle([(250, 250), (250, 350), (350, 350), (350, 250)])
        boundary = Boundary([(0, 0), (0, 500), (500, 500), (500, 0)])
        target = Target([250, 400])
        return agent, target, boundary, obstacle

    env_name = "DeepRL-v0"
    env_args = {"generate_env": env_components}

    env = gym.make(env_name, **env_args)
    vec_env = make_vec_env(
        env_name, n_envs=4, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs=env_args
    )

    model = DQN(
        "MultiInputPolicy",
        vec_env,
        gamma=0.98,
        learning_starts=10000,
        target_update_interval=2000,
        exploration_fraction=0.5,
        verbose=1,
        policy_kwargs={"net_arch": [16, 16]},
    )
    model.learn(total_timesteps=500000, log_interval=4, progress_bar=True)
    model.save("my_model")

    model = DQN.load("my_model", env=vec_env)
    cont = True
    while cont:
        obs = env.reset()
        cumulative_reward = 0
        for i in range(0, 1000):
            print(obs)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            cumulative_reward += reward
            print(f"reward = {reward}, cumilative_reward = {cumulative_reward}")
            env.render()
            if done:
                # cont = False
                break
