from gym.envs.registration import register

register(
    id='Group24M4-v0',
    # entry_point='env.group24_env:Group24',
    entry_point='env.simple_env_wofly_wohc:Group24',
    max_episode_steps=1600,
)