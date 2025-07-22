import json
import numpy as np
from os.path import join
import pdb

from diffuser.guides.policies import Policy
import diffuser.datasets as datasets
import diffuser.utils as utils
from tqdm import tqdm
#随机起点随机终点进行规划

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
    #dataset: str = 'maze2d-large-nowall-v1'
    config: str = 'config.maze2d_cover'
    exp: str = 'cover_nowall'
    renderer: str  = 'utils.Maze2dRendererCover'

#---------------------------------- setup ----------------------------------#

args = Parser().parse_args('plan')

# logger = utils.Logger(args)

#env = datasets.load_environment(args.dataset)
env = datasets.load_environment('maze2d-large-nowall-v1')

#---------------------------------- loading ----------------------------------#

diffusion_experiment = utils.load_diffusion(args.logbase, args.dataset, args.diffusion_loadpath, epoch=args.diffusion_epoch)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
#renderer = diffusion_experiment.renderer
render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    #env=args.dataset,
    env = 'maze2d-large-nowall-v1',
)

renderer = render_config()
policy = Policy(diffusion, dataset.normalizer)

epsilon = 0.1
ratio = 1

def loop(episode=0, plot = False):
    #---------------------------------- one loop ----------------------------------#
    actions_rollout = np.empty((0, 2), dtype=np.float32)
    observation = env.reset()

    print(f'Environment: {args.dataset} | {env.unwrapped.spec.id}')
    print(observation)

    if args.conditional:
        print('Resetting target')
        env.set_target()
        if np.random.rand() < ratio: #强制以一定概率在墙里
            goal = env.unwrapped._target
                
            while not ( goal[0] > 1.5 and goal[0] < 2.5 and goal[1] > 1.5 and goal[1] < 3.5):
                env.set_target()
                goal = env.unwrapped._target

    ## set conditioning xy position to be the goal
    #env._target = np.array([2, 2])  # [x, y] for maze2d
    ## randomly set the target between the walls
    '''target_x = np.random.uniform(1.75, 2.25)  #(2.5, 3.5) #
    target_y = np.random.uniform(2.25, 2.75)    #(2.5, 3.5) #
    env._target = np.array([target_x, target_y])  # [x, y] for maze2d'''
    target = env._target

    ## manually set the target to be the goal
    #target = np.array([2,2])  # [x, y, vx, vy]
    #print(f'Target: {target}')
    cond = {
        diffusion.horizon - 1: np.array([*target, 0, 0]),
    }

    ## observations for rendering
    rollout = [observation.copy()]

    total_reward = 0

    wall = np.array([[1.5,1.5],[1.5,3.5],[2.5,1.5],[2.5,3.5]])
    for t in range(384):

        state = env.state_vector().copy()

        ## can replan if desired, but the open-loop plans are good enough for maze2d
        ## that we really only need to plan once
        if t == 0:
            cond[0] = observation

            action, samples = policy(cond, batch_size=args.batch_size)
            actions = samples.actions[0]
            sequence = samples.observations[0]
            #print(actions.shape, sequence.shape)
            #break
        # pdb.set_trace()


        # ####
        if t < len(sequence) - 1:
            next_waypoint = sequence[t+1]
        else:
            # origin
            #next_waypoint = sequence[-1].copy()
            #next_waypoint[2:] = 0
            # target_orient
            next_waypoint = np.array([target[0], target[1], 0, 0])
            #break
            # pdb.set_trace()

        ## can use actions or define a simple controller based on state predictions
        action_rollout = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
        #print(f't: {t} | action_rollout: {action_rollout} | pos_diff {next_waypoint[:2]-state[:2]} | v_diff {next_waypoint[2:]-state[2:]}' )
        #action_rollout = actions[t+1]
        actions_rollout = np.vstack((actions_rollout, action_rollout))
        '''if t>= 1:
            break
        action = actions[t]  # use the action from the policy'''
        
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



        next_observation, reward, terminal, _ = env.step(action_rollout)
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            if np.linalg.norm(xy - goal) < epsilon:
                #print(f'maze | pos: {xy} | goal: {goal} | HIT GOAL')
                terminal = True

        

        '''if 'maze2d' in args.dataset:
            xy = next_observation[:2]
            goal = env.unwrapped._target
            print(
                f'maze | pos: {xy} | goal: {goal}'
            )
            if xy[0] > 1.5 and xy[0] < 2.5 and xy[1] > 1.5 and xy[1] < 3.5:
                print('HIT WALL')
                terminal = True
                #break
        print(
            f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
            f'{action} | terminal: {terminal} | '
        )'''
        ## update rollout observations
        rollout.append(next_observation.copy())

        # logger.log(score=score, step=t)

        #if t % args.vis_freq == 0 or terminal:
        if (t == env.max_episode_steps - 1 or t==0 or terminal) and plot:
        #if t== len(sequence) - 2 or t==0 or terminal:
            #if episode % 100 == 0:

            fullpath = join(args.savepath, f'{t}_{episode}.png')

            if t == 0: renderer.composite(fullpath, samples.observations, ncol=1)


            # renderer.render_plan(join(args.savepath, f'{t}_plan.mp4'), samples.actions, samples.observations, state)

            ## save rollout thus far
            renderer.composite(join(args.savepath, f'rollout_{episode}.png'), np.array(rollout)[None], ncol=1)

            # renderer.render_rollout(join(args.savepath, f'rollout.mp4'), rollout, fps=80)

            # logger.video(rollout=join(args.savepath, f'rollout.mp4'), plan=join(args.savepath, f'{t}_plan.mp4'), step=t)

        if terminal:
            break

        observation = next_observation
    rollout = np.array(rollout)
    # logger.finish(t, env.max_episode_steps, score=score, value=0)
    #print(rollout.shape, actions_rollout.shape)
    #return rollout, actions_rollout, sequence, actions, target, terminal
    ## save result as a json file
    return rollout, sequence
#loop()
rollouts = []
sequences = []
total_num = 0
total_episode = 20
valid_episode = 0 #only save episodes which arrive at the target
while valid_episode < total_episode:
    rollout, sequence= loop(valid_episode)
    rollouts.append(rollout)
    sequences.append(sequence)
    total_num += len(rollout)
    print(f'Episode {valid_episode} finished with {len(rollout)} rollout steps.')
    valid_episode += 1
print(f'Total length of sequences: {total_num}')
renderer.composite(join(args.savepath, f'rollouts_cover_{total_episode}.png'), rollouts, ncol=1)
renderer.composite(join(args.savepath, f'sequences_cover_{total_episode}.png'), sequences, ncol=1)



