import diffuser.utils as utils
import pdb


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#
#  This script is used to train a cost guide model for diffuser in the maze2d environment. 
class Parser(utils.Parser):
    dataset: str = 'maze2d-large-v1'
    config: str = 'config.maze2d_cost'
    exp: str = 'cost'   # experiment name to specify the save path

diffusion_args = Parser().parse_args('diffusion')
cost_args = Parser().parse_args('cost')



#print(args)

#-----------------------------------------------------------------------------#
#---------------------------------- dataset ----------------------------------#
#-----------------------------------------------------------------------------#

dataset_config = utils.Config(
    args.loader,
    savepath=(args.savepath, 'dataset_config.pkl'),
    env=args.dataset,
    horizon=args.horizon,
    normalizer=args.normalizer,
    preprocess_fns=args.preprocess_fns,
    use_padding=args.use_padding,
    max_path_length=args.max_path_length,
    finetune=args.finetune,
)
dataset = dataset_config()
#print(len(dataset), 'episodes in dataset')


render_config = utils.Config(
    args.renderer,
    savepath=(args.savepath, 'render_config.pkl'),
    env=args.dataset,
)

dataset = dataset_config()
renderer = render_config()

observation_dim = dataset.observation_dim
action_dim = dataset.action_dim


#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#

model_config = utils.Config(
    args.model,
    savepath=(args.savepath, 'model_config.pkl'),
    horizon=args.horizon,
    transition_dim=observation_dim + action_dim,
    cond_dim=observation_dim,
    dim_mults=args.dim_mults,
    device=args.device,
)
cost_model_config = utils.Config(
    cost_args.cost_model,
    savepath = (cost_args.savepath, 'cost_model_config.pkl'),
    input_dim = observation_dim + neiborghor_nums,
    output_dim = observation_dim,
    hidden_dim = cost_args.hidden_dim,
    device = cost_args.device, 
)

diffusion_config = utils.Config(
    args.diffusion,
    savepath=(args.savepath, 'diffusion_config.pkl'),
    horizon=args.horizon,
    observation_dim=observation_dim,
    action_dim=action_dim,
    n_timesteps=args.n_diffusion_steps,
    loss_type=args.loss_type,
    clip_denoised=args.clip_denoised,
    predict_epsilon=args.predict_epsilon,
    ## loss weighting
    action_weight=args.action_weight,
    loss_weights=args.loss_weights,
    loss_discount=args.loss_discount,
    device=args.device,
)
cost_config = utils.Config(
    args.cost,
    savepath = (args.savepath, 'cost_config.pkl'),
)
trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    ema_decay=args.ema_decay,
    sample_freq=args.sample_freq,
    save_freq=args.save_freq,
    label_freq=int(args.n_train_steps // args.n_saves),
    save_parallel=args.save_parallel,
    results_folder=args.savepath,
    bucket=args.bucket,
    n_reference=args.n_reference,
    n_samples=args.n_samples,
)  

policy_config = utils.Config(
    args.poicy,
    savepath=(args.savepath, 'policy_config.pkl'),
    normalizer = dataset.normalizer, 
    sample_fn = n_step_guided_p_sample,
    scale=args.scale,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
    return_diffusion = args.return_diffusion.

)

#-----------------------------------------------------------------------------#
#-------------------------------- instantiate --------------------------------#
#-----------------------------------------------------------------------------#

model = model_config()
cost_model = cost_model_config()

cost = cost_config(cost_model)
diffusion = diffusion_config(model)
policy = policy_config(guide= cost, diffusion_model = diffusion)
#diffusion.load_state_dict(args.diffusion_state_dict)

trainer = trainer_config(diffusion, dataset, renderer)
if args.loadpath is not None:
    trainer.load_from_pt_file(args.loadpath)


#-----------------------------------------------------------------------------#
#------------------------ test forward & backward pass -----------------------#
#-----------------------------------------------------------------------------#

utils.report_parameters(model)

print('Testing forward...', end=' ', flush=True)
batch = utils.batchify(dataset[0])
loss, _ = diffusion.loss(*batch)
loss.backward()
print('✓')


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

for i in range(n_epochs):
    print(f'Epoch {i} / {n_epochs} | {args.savepath}')
    trainer.train(n_train_steps=args.n_steps_per_epoch)

