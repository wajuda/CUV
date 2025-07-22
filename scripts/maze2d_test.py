import os
import collections
import numpy as np
import gym
import pdb

from contextlib import (
    contextmanager,
    redirect_stderr,
    redirect_stdout,
)
#import maze2d
@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def test_maze2d_env():  
    with suppress_output():
        #d4rl prints out a variety of warnings
        import d4rl
    env = gym.make('maze2d-large-v1')
    #env = gym.make('maze2d-large-nowall-v1')
    '''from d4rl.pointmaze.maze_model import MazeEnv
    from gym.envs.registration import register
    LARGE_MAZE_nowall = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#OOOO#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"
    register(
    id='maze2d-large-nowall-v1',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_MAZE_nowall,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 6.7,
        'ref_max_score': 273.99,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
    }
)
    env = gym.make('maze2d-large-nowall-v1')'''
    env.reset()
    #print(env.max_episode_steps)
    if hasattr(env, 'grid'):
        print("迷宫地图（墙=1）：\n", env.grid)

    print("\n=== Attributes/Methods ===")
    public_attrs = [attr for attr in dir(env) if attr.startswith('_')]
    print(public_attrs)
    print("\n=== Public Attributes/Methods ===")
    public_attrs = [attr for attr in dir(env) if not attr.startswith('_')]
    print(public_attrs)
    for _ in range(100):
        action = env.action_space.sample()  # Sample a random action
        obs, reward, done, info = env.step(action)  # Take a step in the environment
        if done:
            env.reset()  # Reset the environment if done
    env_unwrapped = env.unwrapped
    if hasattr(env_unwrapped, 'grid'):
        print("迷宫地图（墙=1）：\n", env_unwrapped.grid)
    else:
        print("The unwrapped environment does not have a grid attribute.")
    print("\n=== Attributes/Methods ===")
    public_attrs = [attr for attr in dir(env_unwrapped)]
    print(public_attrs)
    possible_maze_attrs = ["maze_arr", "str_maze_spec", "empty_and_goal_locations", "goal_locations"]
    for attr in possible_maze_attrs:
        if hasattr(env_unwrapped, attr):
            print(f"{attr}: {getattr(env_unwrapped, attr)}")
        else:
            print(f"{attr} not found")
    env.close()
    assert True  # If no exceptions were raised, the test passes
if __name__ == "__main__":
    #print(list(gym.envs.registry.keys()))
    test_maze2d_env()
    print("Maze2D environment test passed successfully.")
# This code tests the maze2d environment by creating an instance of it,
# resetting it, and taking random actions for 100 steps.
# If no exceptions are raised during this process, the test passes.