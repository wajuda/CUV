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

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl
from d4rl.pointmaze.maze_model import MazeEnv
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
LARGE_MAZE_block = \
    "############\\"+\
    "#OOOO#OOOOO#\\"+\
    "#O##O#O#O#O#\\"+\
    "#OOOOOO#OOO#\\"+\
    "#O########O#\\"+\
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
register(
    id='maze2d-large-block-v1',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=800,
    kwargs={
        'maze_spec':LARGE_MAZE_block,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 6.7,
        'ref_max_score': 273.99,
        'dataset_url':'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
    }
)

#-----------------------------------------------------------------------------#
#-------------------------------- general api --------------------------------#
#-----------------------------------------------------------------------------#

def load_environment(name):
    if type(name) != str:
        ## name is already an environment
        print(f'Using environment {name}')
        return name
    print(f'Loading environment {name}')
    with suppress_output():
        wrapped_env = gym.make(name)
    env = wrapped_env.unwrapped
    env.max_episode_steps = wrapped_env._max_episode_steps
    env.name = name
    return env

def get_dataset(env):
    dataset = env.get_dataset()

    if 'antmaze' in str(env).lower():
        ## the antmaze-v0 environments have a variety of bugs
        ## involving trajectory segmentation, so manually reset
        ## the terminal and timeout fields
        dataset = antmaze_fix_timeouts(dataset)
        dataset = antmaze_scale_rewards(dataset)
        get_max_delta(dataset)

    return dataset
 
def sequence_dataset_from_npz_file(file_name = '/home/junda/diffuser/logs/maze2d-large-v1/plans/release_H384_T256_LimitsNormalizer_b1_condFalse_pFalse/0/all_data_episode.npz'):
    """
    Returns an iterator through trajectories from a npz file.
    Args:
        file_name: The path to the npz file containing the dataset.
        all_data.update({
            f'loop_{valid_episode}/actions': valid_actions,
            f'loop_{valid_episode}/observations': valid_sequence,
        })
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
    """
    print(f'[ datasets/npz ] Loading dataset from {file_name}')
    data = np.load(file_name, allow_pickle=True)
    
    # Get all episode keys (e.g., 'loop_0', 'loop_1', etc.)
    episode_keys = [key.split('/')[0] for key in data.files if '/actions_rollout' in key or '/rollout' in key]
    episode_keys = sorted(list(set(episode_keys)), key=lambda x: int(x.split('_')[1]))
    
    num_episodes = len(episode_keys)

    print(f'[ datasets/npz ] Dataset has {num_episodes} episodes')
    
    for ep_key in episode_keys:
        # Get observations and actions for this episode
        obs_key = f'{ep_key}/rollout'
        act_key = f'{ep_key}/actions_rollout'
        
        observations = data[obs_key]
        actions = data[act_key]
        terminals = np.zeros(len(observations), dtype=bool)  # Dummy terminals
        timeouts = np.zeros(len(observations), dtype=bool)  # Dummy timeouts
        terminals[-1] = True  # Mark the last observation as terminal
        timeouts[-1] = True  # Mark the last observation as timeout
        
        # Create episode dictionary to match sequence_dataset format
        episode_data = {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'terminals': terminals,
            'timeouts': timeouts,
            # Add dummy rewards and terminals for compatibility
            #'rewards': np.zeros(len(observations)),
            #'terminals': np.zeros(len(observations), dtype=bool),
        }
        
        # For maze2d environments, we might need additional processing
        #if 'maze2d' in file_name:
            #episode_data = process_maze2d_episode(episode_data)
        episode_data = process_maze2d_episode(episode_data)
        yield episode_data
    
    print(f'[ datasets/npz ] Processed {num_episodes} episodes from file')
    

def sequence_dataset(env, preprocess_fn):
    """
    Returns an iterator through trajectories.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        An iterator through dictionaries with keys:
            observations
            actions
            rewards
            terminals
    """
    dataset = get_dataset(env)
    dataset = preprocess_fn(dataset)

    N = dataset['rewards'].shape[0]
    print(f'[ step0 datasets/d4rl ] Dataset has {N} length')
    data_ = collections.defaultdict(list)
    for k in dataset:
        print(k)

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = 'timeouts' in dataset
    num_episodes  = 0
    episode_step = 0
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)

        for k in dataset:
            if 'metadata' in k: continue
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            num_episodes += 1
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            if 'maze2d' in env.name:
                episode_data = process_maze2d_episode(episode_data)
            yield episode_data
            data_ = collections.defaultdict(list)

        episode_step += 1
    print(f'[ step1 datasets/d4rl ] Found {num_episodes} episodes in dataset')


#-----------------------------------------------------------------------------#
#-------------------------------- maze2d fixes -------------------------------#
#-----------------------------------------------------------------------------#

def process_maze2d_episode(episode):
    '''
        adds in `next_observations` field to episode
    '''
    assert 'next_observations' not in episode
    length = len(episode['observations'])
    next_observations = episode['observations'][1:].copy()
    for key, val in episode.items():
        episode[key] = val[:-1]
    episode['next_observations'] = next_observations
    return episode

if __name__ == '__main__':
    print('Loading new dataset')
    itr = sequence_dataset_from_npz_file()
    for episode in itr:
        print(episode['observations'].shape, episode['actions'].shape)
