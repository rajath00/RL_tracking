from gym.envs.registration import register

register(
    id="DeepRL-v0",
    entry_point="Own_gym.rl_env:Own_gym",
    max_episode_steps=1000,
)