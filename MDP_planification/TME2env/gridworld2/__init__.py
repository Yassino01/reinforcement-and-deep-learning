from gridworld2.gridworld_env import GridworldEnv

from gym.envs.registration import register

register(
    id='gridworld-v1',
    entry_point='gridworld2:GridworldEnv',
)
