import json
import numpy as np
from os.path import join
import pdb 
import torch

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from tqdm import tqdm

def get_valid_new_path(path, actions, ratio=0.9):
    length = len(path)
    #start_point = path[0]
    target_point = path[-1]
    max_distance = 0
    num_back = 0
    j = -1
    for i in range(2, length):
        current_point = path[-i]
        distance = np.linalg.norm(current_point - target_point)
        if distance > ratio*max_distance:
            if distance > max_distance:
                max_distance = distance
            num_back = 0
            j= -i
        else:
            num_back += 1
        if num_back > 30:
            break
    
    return path[j:], actions[j:]

class Parser(utils.Parser):
    #dataset: str = 'maze2d-umaze-v1'
    dataset: str = 'maze2d-large-v1'
    real_dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d_action'
    exp: str = 'guided_test'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)
env = datasets.load_environment(args.real_dataset)

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

epsilon = 0.1
from diffuser.guides.guides import ValueGuide_maze2d

value_guide = ValueGuide_maze2d(
    #normalizer=dataset.normalizer,
    maze_layout=env.unwrapped.str_maze_spec
)

'''[[0.0, 0.0, 7.0, 2.0]
                , [0.0, 0.0, 6.0, 2.3]
                , [0.0, 0.0, 5.0, 2.2]
                , [0.0, 0.0, 4.7, 1.8]
                , [ 0.0, 0.0, 4.8, 0.9]
                , [0.0, 0.0, 3.8, 0.9]]'''
traj = torch.tensor([[7.0, 2.0, 0.0, 0.0]
                , [6.0, 2.3, 0.0, 0.0]
                , [5.0, 2.2, 0.0, 0.0]
                , [4.7, 1.8, 0.0, 0.0]
                , [4.8, 0.9, 0.0, 0.0]
                , [3.8, 0.9, 0.0, 0.0]]
                , dtype=torch.float32).unsqueeze(0)  # [1, T, 4]
#traj = traj.to(diffusion.device)
print(traj.shape)
print( np.array(traj)[0][:,:2].shape)
renderer.composite(join(args.savepath, f'0.png'), np.array(traj)[:,:,:2], ncol=1)

max_iter = 10
learning_rate = 1
for i in range(max_iter):
    traj.requires_grad_(True)
    values, grad = value_guide.gradients(traj)
    traj = traj.detach() + learning_rate * grad
    renderer.composite(join(args.savepath, f'{i+1}.png'), np.array(traj)[:,:,:2], ncol=1)


