from gym.envs.registration import register

register(
    id='goleft-v0',
    entry_point='custom_envs.envs.GoLeft:GoLeftEnv'
)
