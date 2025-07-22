import json
import numpy as np
from os.path import join
import pdb 

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
    exp: str = 'start502goal710'
 
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

def loop(episode=0):
    #---------------------------------- one loop ----------------------------------#
    actions_rollout = np.empty((0, 2), dtype=np.float32)
    #observation = env.reset()
    observation = env.reset_to_location(np.array([7, 1,7]))  # [x, y, vx, vy]

    print(f'Environment: {args.dataset} | {env.unwrapped.spec.id}')
    print(observation)

    if args.conditional:
        print('Resetting target')
        env.set_target()

    ## set conditioning xy position to be the goal
    #env._target = np.array([2, 2])  # [x, y] for maze2d
    ## randomly set the target between the walls
    '''target_x = np.random.uniform(1.75, 2.25)  #(2.5, 3.5) #
    target_y = np.random.uniform(2.25, 2.75)    #(2.5, 3.5) #
    env._target = np.array([target_x, target_y])  # [x, y] for maze2d'''
    target_x = np.random.uniform(2.8, 3.2)  #(2.5, 3.5) #
    target_y = np.random.uniform(1.8, 2.2)    #(2.5, 3.5) #
    env._target = np.array([target_x, target_y])  # [x, y] for maze2d
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
    for t in range(env.max_episode_steps):

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
        '''if t<len(sequence):
            action_rollout = actions[t]
        else:
            terminal=True
            break'''
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
                print(f'maze | pos: {xy} | goal: {goal} | HIT GOAL')
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
        #if t == env.max_episode_steps - 1 or t==0 or terminal:
        if t== len(sequence) - 2 or t==0 or terminal:
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
    return rollout, actions_rollout, sequence, actions, target, terminal
    ## save result as a json file
    '''json_path = join(args.savepath, 'rollout.json')
    json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)'''
    ''' # set keys to dicede which keys to save in json
    json_path = join(args.savepath, 'rollout.json')
    json_data = {
        'score': score,
        'step': t,
        'return': total_reward,
        'term': terminal,
        'epoch_diffusion': diffusion_experiment.epoch,
        'target': target.tolist(),
        'actions': actions.tolist(),
        'sequence': sequence.tolist(),
    }
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
    print(f'Saved rollout to {json_path}')
    print(f'Finished loop with score: {score:.4f} | total reward: {total_reward:.2f} | steps: {t}')'''
#loop()
all_data = {}
total_num = 0
total_episode = 10
valid_episode = 0 #only save episodes which arrive at the target
while valid_episode < total_episode:
    rollout, actions_rollout, sequence, actions, target, terminal= loop(valid_episode)
    if terminal:
        
        valid_rollout, valid_actions_rollout = get_valid_new_path(rollout, actions_rollout)
        valid_sequence, valid_actions = get_valid_new_path(sequence, actions)
        all_data.update({
            f'loop_{valid_episode}/rollout': valid_rollout,
            f'loop_{valid_episode}/actions_rollout': valid_actions_rollout,
            f'loop_{valid_episode}/actions': valid_actions,
            f'loop_{valid_episode}/sequence': valid_sequence,
            f'loop_{valid_episode}/target': target,
        })
        renderer.composite(join(args.savepath, f'valid_rollout_{valid_episode}.png'), np.array(valid_rollout)[None], ncol=1)
        renderer.composite(join(args.savepath, f'valid_sequence_{valid_episode}.png'), np.array(valid_sequence)[None], ncol=1)
        total_num += valid_actions_rollout.shape[0]
        total_num += valid_actions.shape[0]
        print(f'Rollout: {rollout.shape[0]}, Actions: {actions_rollout.shape[0]}, Actions: {actions.shape[0]}, Sequence: {sequence.shape[0]}, Target: {target}')
        print(f'valid_Rollout: {valid_rollout.shape[0]}, valid_Actions_rollout: {valid_actions_rollout.shape[0]}, valid_Actions: {valid_actions.shape[0]}, valid_Sequence: {valid_sequence.shape[0]}, Target: {target}')
        print(f'Finished loop {valid_episode+1}/{total_episode}')
        valid_episode += 1
    else:
        print(f'Loop {valid_episode+1}/{total_episode} did not reach the target, retrying...')
print(f'Total length of path: {total_num}')
np.savez_compressed(
    join(args.savepath, f'all_data_{total_episode}.npz'),
    **all_data
)
print(f'Saved all data to {join(args.savepath, "all_data.npz")}')

