import diffuser.utils as utils
import pdb
import diffuser.datasets as datasets
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
from diffuser.models.diffusion import n_step_guided_p_sample, default_sample_fn
import numpy as np
 



#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
#  This script is used to train a cost guide model for diffuser in the maze2d environment.
# the robust version means interact with no guide, then train cost to  convergence, finally test with different guide scale.
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    real_dataset: str = 'maze2d-large-block-v1'
    config: str = 'config.maze2d_cost_robust'
    #exp: str = 'cost_robust_debug'   # experiment name to specify the save path

diffusion_args = Parser().parse_args('diffusion')
cost_args = Parser().parse_args('cost')



#-----------------------------------------------------------------------------#
#----------------------------------- logger & writer -----------------------------------#
#-----------------------------------------------------------------------------#
writer = SummaryWriter(log_dir = diffusion_args.savepath)
logger = logging.getLogger("train_cost")
logger.setLevel(logging.INFO)

# 2. 配置日志输出到文件
file_handler = logging.FileHandler(diffusion_args.savepath + '/test.log', mode = 'w')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()  # 同时输出到控制台


file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
#assert False, f'{diffusion_args.exp_name}'



#-----------------------------------------------------------------------------#
#---------------------------------- env ----------------------------------#
#-----------------------------------------------------------------------------#
env = datasets.load_environment(diffusion_args.real_dataset)
logger.info(f"Environment {diffusion_args.real_dataset} loaded successfully.")



#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    diffusion_args.loader,
    savepath=(diffusion_args.savepath, 'dataset_config.pkl'),
    env=diffusion_args.real_dataset,
    horizon=diffusion_args.horizon,
    normalizer=diffusion_args.normalizer,
    preprocess_fns=diffusion_args.preprocess_fns,
    use_padding=diffusion_args.use_padding,
    max_path_length=diffusion_args.max_path_length,
    #finetune=diffusion_args.finetune,
)
dataset = dataset_config()
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim
logger.info(f"Dataset {diffusion_args.real_dataset} loaded successfully with observation_dim={observation_dim}, action_dim={action_dim}.")

#-----------------------------------------------------------------------------#
#---------------------------------- render ----------------------------------#
#-----------------------------------------------------------------------------#


render_config = utils.Config(
    diffusion_args.renderer,
    savepath=(diffusion_args.savepath, 'render_config.pkl'),
    env=diffusion_args.real_dataset,
)



#-----------------------------------------------------------------------------------------#
#------------------------------ diffusion model & cost model------------------------------#
#-----------------------------------------------------------------------------------------#

diffusion_model_config = utils.Config(
    diffusion_args.model,
    savepath=(diffusion_args.savepath, 'model_config.pkl'),
    horizon=diffusion_args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=diffusion_args.dim_mults,
    device=diffusion_args.device,
)
cost_model_config = utils.Config(
    cost_args.model,
    savepath = (diffusion_args.savepath, 'cost_model_config.pkl'),
    input_dim = 2*observation_dim + cost_args.neighbour_num,
    output_dim = 1,
    hidden_dim = cost_args.hidden_dim,
    device = cost_args.device, 
)

diffusion_config = utils.Config(
    diffusion_args.diffusion,
    savepath=(diffusion_args.savepath, 'diffusion_config.pkl'),
    horizon=diffusion_args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=diffusion_args.n_diffusion_steps,
    loss_type=diffusion_args.loss_type,
    clip_denoised=diffusion_args.clip_denoised,
    predict_epsilon=diffusion_args.predict_epsilon,
    ## loss weighting
    action_weight=diffusion_args.action_weight,
    loss_weights=diffusion_args.loss_weights,
    loss_discount=diffusion_args.loss_discount,
    device=diffusion_args.device,
)
cost_config = utils.Config(
    cost_args.cost,
    savepath = (diffusion_args.savepath, 'cost_config.pkl'),
    maze_layout=env.unwrapped.str_maze_spec, 
    #maze_layout = maze_spec,
    normalizer=dataset.normalizer,
    neighbour_num = cost_args.neighbour_num,
    loss_type = diffusion_args.loss_type,
)


#-----------------------------------------------------------------------------#
#------------------------------ buffer------------------------------#
#-----------------------------------------------------------------------------#
buffer_config = utils.Config(
    cost_args.buffer,
    savepath = (diffusion_args.savepath,'buffer_config.pkl'),
    buffer_size = cost_args.buffer_size,
)
 
#-----------------------------------------------------------------------------#
#------------------------------ policy & trainer------------------------------#
#-----------------------------------------------------------------------------#
policy_config = utils.Config(
    diffusion_args.policy,
    savepath=(diffusion_args.savepath, 'policy_config.pkl'),
    normalizer = dataset.normalizer, 
    sample_fn = n_step_guided_p_sample,#diffusion_args.sample_fn,
    scale=diffusion_args.scale,
    n_guide_steps=diffusion_args.n_guide_steps,
    t_stopgrad=diffusion_args.t_stopgrad,
    scale_grad_by_std=diffusion_args.scale_grad_by_std,
    verbose=False,
    return_diffusion = diffusion_args.return_diffusion,

)



trainer_config = utils.Config(
    diffusion_args.trainer,
    logger = logger,
    savepath=(diffusion_args.savepath, 'trainer_config.pkl'),
    train_batch_size=diffusion_args.batch_size,
    train_lr=diffusion_args.learning_rate,
    #gradient_accumulate_every=diffusion_args.gradient_accumulate_every,
    #ema_decay=diffusion_args.ema_decay,
    conditional = diffusion_args.conditional,
    #sample_freq=diffusion_args.sample_freq,
    save_freq=diffusion_args.save_freq,
    #label_freq=int(diffusion_args.n_train_episodes // diffusion_args.n_saves),
    save_parallel=diffusion_args.save_parallel,
    results_folder=diffusion_args.savepath,
    bucket=diffusion_args.bucket,
    epsilon = diffusion_args.epsilon,
    #update_guide_freq = diffusion_args.update_guide_freq,
    sample_batch_size = diffusion_args.sample_batch_size,
    n_collect_episodes = diffusion_args.n_collect_episodes,
    n_train_epochs = diffusion_args.n_train_epochs,
    vis_test_freq = diffusion_args.vis_test_freq,
    vis_collect_freq = diffusion_args.vis_collect_freq,
    #n_reference=diffusion_args.n_reference,
    #n_samples=diffusion_args.n_samples,
)  


'''guide_config = utils.Config(
    'guides.guides.ValueGuide_maze2d',
    maze_layout=env.unwrapped.str_maze_spec, 
    #maze_layout = maze_spec,
    normalizer=dataset.normalizer,
)

guide = guide_config()'''
#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#
 
diffusion_model = diffusion_model_config()
cost_model = cost_model_config()

cost = cost_config(cost_model)
diffusion = diffusion_config(diffusion_model)
policy = policy_config(guide= cost, diffusion_model = diffusion)
#
#diffusion.load_state_dict(args.diffusion_state_dict)
# baseline_policy  no guide
baseline_policy_config = utils.Config(
    diffusion_args.policy,
    savepath=(diffusion_args.savepath, 'baseline_policy_config.pkl'),
    normalizer = dataset.normalizer, 
    sample_fn = default_sample_fn,#diffusion_args.sample_fn,
    verbose=False,
    return_diffusion = diffusion_args.return_diffusion,
)
baseline_diffusion_model = diffusion_model_config()
baseline_diffusion = diffusion_config(baseline_diffusion_model)
baseline_policy = baseline_policy_config(diffusion_model = baseline_diffusion)
if diffusion_args.loadpath is not None:
    #assert False, "Loading from a path is not supported in cost training."
    data = torch.load(diffusion_args.loadpath)
        #self.step = data['step']
    baseline_policy.diffusion_model.load_state_dict(data['model'])
    logger.info(f"Baseline Diffusion model loaded from {diffusion_args.loadpath} successfully.")

if cost_args.loadpath is not None:
    data = torch.load(cost_args.loadpath)
    policy.guide.load_state_dict(data['cost_model'])
    logger.info(f"Cost model loaded from {cost_args.loadpath} successfully.")


renderer = render_config()
buffer = buffer_config()

trainer = trainer_config(env= env, renderer = renderer, policy = policy, baseline_policy = baseline_policy, buffer = buffer, writer = writer, dataset = dataset)
if diffusion_args.loadpath is not None:
    #assert False, "Loading from a path is not supported in cost training."
    trainer.load_from_pt_file(diffusion_args.loadpath)
    #logger.info(f"Diffusion model loaded from {diffusion_args.loadpath} successfully.")




#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(diffusion_model)   # 3.68M
#utils.report_parameters(cost_model)   #9.6k
#logger.info(f"Diffusion model and cost model initialized with {utils.count_parameters(diffusion_model)} and {utils.count_parameters(cost_model)} parameters respectively.")




#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#


# collect dataset
#trainer.collect()

# train cost model
#trainer.train()

# test with diffetent scale
#scales = cost_args.scales

#------------random test---------------------
scales = cost_args.scales
scores = []
rewards = []
for i in range(diffusion_args.n_test_samples):
    for scale in scales:
        trainer.test(scale = scale, episode = i)
    scores.append(trainer.scores)
    rewards.append(trainer.rewards)

for i, scale in enumerate(scales):
    logger.info(f'scale{scale} mean-score: {np.array(scores)[:,i].mean()} mean_reward: {np.array(rewards)[:,i].mean()}')


#-------------task test
'''scales = [0, 100, 10000 ,1000000, 100000000, 100000000000000000]
scores = []
rewards = []
starts = [[1,1],[7,2],[1,9]]
target = [7.0, 10.0]
trainer.vis_test_freq = 1
#for i in range(diffusion_args.n_test_samples):
for i, start in enumerate(starts):
    for scale in scales:
        trainer.test_task(scale = scale, episode = i, start = start, target= target)
    scores.append(trainer.scores)
    rewards.append(trainer.rewards)

for i, scale in enumerate(scales):
    logger.info(f'scale{scale} mean-score: {np.array(scores)[:,i].mean()} mean_reward: {np.array(rewards)[:,i].mean()}')'''