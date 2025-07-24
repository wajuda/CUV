import os
import copy
import numpy as np
import torch
import einops
import pdb
 
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch)

                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            self.step += 1

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def load_from_pt_file(self, loadpath):
        data = torch.load(loadpath)
        #self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema']) 
        print(f'[ utils/training ] Loaded model from {loadpath}')

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)


# This class is used to train a cost guide model for diffuser in the maze2d environment.
class CostTrainer(object):                                                              
    def __init__(
        self,
        #diffusion_model,
        #cost_model, 
        env = None,   #   for interaction with  the environment
        dataset = None,  #
        renderer = None, #for rendering the trajectory
        policy = None,  # including the cost, diffusion, normalizer,
        buffer = None, # the experience can be reused, because i want to use the relative position rather than the absolute position
        conditional = True,
        writer = None,
        logger = None,
        epsilon=0.1, # when planed position is far from the real position, we need to replan
        #ema_decay=0.995,  # do not use ema first, because it is just a simple mlp model for cost mapping
        sample_batch_size=10, # sample the highest value within sample_batch_size traj. 
        train_batch_size=32,
        train_lr=2e-5,
        test_freq = 5,
        #log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        #n_reference=8,
        #n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.policy = policy
        #self.cost_model = policy.guide
        self.diffusion_model = policy.diffusion_model
        self.writer = writer
        self.logger = logger
        self.buffer = buffer
        #self.ema = EMA(ema_decay)
        #self.ema_cost_model = copy.deepcopy(self.cost_model)
        #self.update_ema_every = update_ema_every
        self.env = env
        self.epsilon = epsilon
        #self.cost_weight = cost_weight
        self.conditional = conditional
        self.test_freq = test_freq
 
        #self.step_start_ema = step_start_ema
        self.sample_batch_size = sample_batch_size
        #self.log_freq = log_freq
        #self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        #self.gradient_accumulate_every = gradient_accumulate_every


        '''self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))'''
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(self.policy.guide.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        #self.n_reference = n_reference
        #self.n_samples = n_samples

        #self.reset_parameters()
        #self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    def replay(self):
        if len(buffer) < self.batch_size:
            return 0
        batch = buffer.sample(self.batch_size)
        batch = batch_to_device(batch)
        loss = self.policy.guide.loss(*batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    def train(self, n_train_episodes):

        #timer = Timer()
        for episode in range(n_train_episodes):
            #记录plan和执行的observation，可视化renderer
            #real_observation = np.zeros([self.env.max_episode_steps+1,self.env.observation_dim], dtype=np.float32)
            #plan_observation = np.zeros([self.env.max_episode_steps+1,self.env.observation_dim], dtype=np.float32)
            #actions_rollout = np.vstack((actions_rollout, action_rollout))
            observation = self.env.reset()
            #real_observation[0] = observation.copy()
            #plan_observation[0] = observation.copy()
            if self.conditional == True:
                self.env.set_target()
            target = self.env.get_target()
            cond = {diffusion.horizon - 1: np.array([*target, 0, 0]),}
            t= 0 #t in the planing    step is for the execution
            total_reward = 0.0
            for step in range(self.env.max_episode_steps):
                if t == 0:
                    cond[0] = observation

                    action, samples = self.policy(cond, batch_size=self.sample_batch_size)
                    actions = samples.actions[0]
                    sequence = samples.observations[0]
                    value = samples.value[0]
                
                if t < len(sequence) - 1:
                    next_waypoint = sequence[t+1]
                else:
                    next_waypoint = sequence[-1].copy()
                    next_waypoint[2:] = 0

                action = next_waypoint[:2] - observation[:2] + (next_waypoint[2:] - observatoon[2:])
                next_observation, reward, terminal, _ = self.env.step(action)
                real_observation[step+1] = next_observation.copy()
                plan_observation[step+1] = next_waypoint.copy()
                x, y = self.policy.guide.get_training_data(observation, next_observation, next_waypoint)
                total_reward += reward
                score = self.env.get_normalized_score(total_reward)
                '''print(
                    f't: {t} | r: {reward:.2f} |  R: {total_reward:.2f} | score: {score:.4f} | '
                    f'{action} | terminal: {terminal} | '
                )'''
                self.buffer.add(
                    data= x, # 4+4+neighbor_num
                    label = y, #-cost
                )
                observation = next_observation
                t += 1
                if torch.norm(observation - next_waypoint) > self.epsilon:
                   t=0
                if step % self.update_guide_freq == 0:
                    loss = self.replay()
                    self.logger.info(f'episode{episode}step{step}loss{loss}')
                    self.writer.add_scalar('loss', loss, episode * self.env.max_episode_steps + step)
            # end of the episode
            # i want to save the model log the loss, reward, score test the model.
            self.logger.info(f'Episode {episode} | Total Reward: {total_reward:.2f} | Score: {score:.4f}')
            self.writer.add_scalar('total reward', total_reward, episode)
            self.writer.add_scalar('score', score, episode)

            #  plot the traj
            '''if episode == 0 and self.sample_freq:
                self.render_reference(self.n_reference) 
            if self.sample_freq and episode % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples) '''

            # save the model  
            if episode % self.save_freq == 0:
                label = episode // self.label_freq * self.label_freq
                self.save(label)    

            # test the mean performance on multiple traj 
            if episode % self.test_freq == 0:
                self.test(episode)

    def test(self, episode):
        with torch.no_grad():
            scores = []
            rewards = []
            for i in range(self.n_test_sampels):
                observation = self.env.reset()
                if self.conditional == True:
                    self.env.set_target()
                target = self.env.get_target()
                cond = {diffusion.horizon - 1: np.array([*target, 0, 0]),}
                total_reward = 0.0
                if i == 0:
                    rollout = [observation.copy()]
                for t in range(self.env.max_episode_steps):
                    if t == 0:
                        cond[0] = observation

                        action, samples = self.policy(cond, batch_size=self.sample_batch_size)
                        actions = samples.actions[0]
                        sequence = samples.observations[0]
                        value = samples.value[0]

                        
                    
                    if t < len(sequence) - 1:
                        next_waypoint = sequence[t+1]
                    else:
                        next_waypoint = sequence[-1].copy()
                        next_waypoint[2:] = 0

                    action = next_waypoint[:2] - observation[:2] + (next_waypoint[2:] - observatoon[2:])
                    next_observation, reward, terminal, _ = self.env.step(action)
                    if i==0:
                        rollout.append(next_observation.copy())
                    #real_observation[step+1] = next_observation.copy()
                    #plan_observation[step+1] = next_waypoint.copy()
                    #x, y = self.policy.guide.get_training_data(observation, next_observation, next_waypoint)
                    total_reward += reward
                    score = self.env.get_normalized_score(total_reward)
                if i==0:
                    savepath = os.path.join(self.logdir, f'rollout{episode}.png')
                    self.renderer.composite(savepath, rollout)
                    savepath = os.path.join(self.logdir, f'plan{episode}.png')
                    self.renderer.composite(savepath, samples.observations)
                scores.append(score)
                rewards.append(total_reward)
            self.logger.info(f'episode{episode} test: mean_score{mean(scores)} mean_reward{mean(rewards)}')
            self.writer.add_scalar('test_mean_score', mean(scores), episode)
            self.writer.add_scalar('test_mean_reward', mean(rewards), episode)
                
                
        

                
 
    ## can use actions or define a simple controller based on state predictions
    #action = next_waypoint[:2] - state[:2] + (next_waypoint[2:] - state[2:])
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'cost_model': self.policy.guide.state_dict(), 
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        self.logger.info(f'[ utils/cost-training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def load_from_pt_file(self, loadpath):
        data = torch.load(loadpath)
        #self.step = data['step']
        self.policy.diffusion_model.load_state_dict(data['model'])
        #self.ema_model.load_state_dict(data['ema']) 
        self.logger.info(f'load diffusion model from {loadpath}')

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = samples[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)



