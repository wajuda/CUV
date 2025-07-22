import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils


class Parser(utils.Parser):
    #dataset: str = 'maze2d-umaze-v1'
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)
'''from maze2d_test import suppress_output
with suppress_output():
        #d4rl prints out a variety of warnings
        import d4rl
    #env = gym.make('maze2d-large-v1')
from d4rl.pointmaze.maze_model import MazeEnv
import gym
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
#env = datasets.load_environment('maze2d-large-nowall-v1')
env = datasets.load_environment(args.dataset)
 
#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

policy = Policy(diffusion, dataset.normalizer)

#---------------------------------- main loop ----------------------------------#

observation = env.reset()

print(f'Environment: {args.dataset} | {env.unwrapped.spec.id}')
print(observation)

if args.conditional:
    print('Resetting target')
    env.set_target()

## set conditioning xy position to be the goal
#env._target = np.array([2,2.5])
target = env._target
## manually set the target to be the goal

#print(f'Target: {target}')
cond = {
    diffusion.horizon - 1: np.array([*target, 0, 0]),
}

## observations for rendering
rollout = [observation.copy()]

total_reward = 0
## wall corner and center for maze2d
wall_corner = np.array([[1.5,1.5],[1.5,3.5],[2.5,1.5],[2.5,3.5]])
wall_center = [[2,2],[2,3]]
for t in range(env.max_episode_steps):

    state = env.state_vector().copy()

    ## can replan if desired, but the open-loop plans are good enough for maze2d
    ## that we really only need to plan once
    if args.replan:
        # replan every step
        cond[0] = observation
        _, samples = policy(cond, batch_size=args.batch_size)
        #actions = samples.actions[0]
        sequence = samples.observations[0]
        next_waypoint = sequence[1]  # use the next waypoint in the sequence
    else:
        if t == 0:
            cond[0] = observation

            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]
        # pdb.set_trace()


        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            next_waypoint = sequence[-1].copy()
            next_waypoint[2:] = 0
            # pdb.set_trace()
    


    ## can use actions or define a simple controller based on state predictions
    #action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    if t>= 384:
        break
    action = actions[t]  # use the action from the policy
    
    # pdb.set_trace()
    ####

    # else:
    #     actions = actions[1:]
    #     if len(actions) > 1:
    #         action = actions[0]
    #     else:
    #         # action = np.zeros(2)
    #         action = -state[2:]
    #         pdb.set_trace()



    next_observation, reward, terminal, _ = env.step(action)
    total_reward += reward
    score = env.get_normalized_score(total_reward)
    print(
        f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
        f'{action} | terminal: {terminal} | '
    )

    if 'maze2d' in args.dataset:
        xy = next_observation[:2]
        goal = env.unwrapped._target
        print(
            f'maze | pos: {xy} | goal: {goal}'
        )
        '''if xy[0] > 1.5 and xy[0] < 2.5 and xy[1] > 1.5 and xy[1] < 3.5:
            print('HIT WALL')
            terminal = True
            break'''
        '''current_grid_x = int(xy[0] + 0.5)
        current_grid_y = int(xy[1] + 0.5)
        if [current_grid_x, current_grid_y] in wall_center:
            print('HIT WALL CENTER')
            terminal = True
            break'''

    ## update rollout observations
    rollout.append(next_observation.copy())

    # logger.log(score=score, step=t)

    if t % args.vis_freq == 0 or terminal:
        fullpath = join(args.savepath, f'{t}.png')

        if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


        # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

        ## save rollout thus far
        renderer.composite(join(args.savepath, 'rollout.png'), np.array(rollout)[None], ncol=1)

        # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

        # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

    if terminal:
        break

    observation = next_observation

# logger.finish(t, env.max_episode_steps, score=score, value=0)

## save result as a json file
json_path = join(args.savepath, 'rollout.json')
json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
    'epoch_diffusion': diffusion_experiment.epoch}
json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
 