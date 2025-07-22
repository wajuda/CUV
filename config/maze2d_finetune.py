import socket

from diffuser.utils import watch

#------------------------ base ------------------------#

## automatically make experiment names for planning
## by labelling folders with these args

diffusion_args_to_watch = [
    ('prefix', ''),
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('exp', 'E'),
]

 
plan_args_to_watch = [
    ('prefix', ''),
    ##
    ('horizon', 'H'),
    ('n_diffusion_steps', 'T'),
    ('value_horizon', 'V'),
    ('discount', 'd'),
    ('normalizer', ''),
    ('batch_size', 'b'),
    ##
    ('conditional', 'cond'),
    ('replan', 'p')
]

base = {

    'diffusion': {
        ## model
        'model': 'models.TemporalUnet',
        'diffusion': 'models.GaussianDiffusion',
        'horizon': 256,
        'n_diffusion_steps': 256,
        'action_weight': 1,
        'loss_weights': None,
        'loss_discount': 1,
        'predict_epsilon': False,
        'dim_mults': (1, 4, 8),
        'renderer': 'utils.Maze2dRenderer',

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 800,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion_finetune/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'n_steps_per_epoch': 800,
        'loss_type': 'l2',
        'n_train_steps': 2e4,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'gradient_accumulate_every': 2,
        'ema_decay': 0.995,
        'save_freq': 5000,
        'sample_freq': 5000,
        'n_saves': 50,
        'save_parallel': False,
        'n_reference': 50,
        'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
        ## loading
        'finetune': True,
        'loadpath': '/home/junda/diffuser/logs/maze2d-large-v1/diffusion/H384_T256/state_1960000.pt',
    },

    'plan': {
        'batch_size': 1,
        'device': 'cuda',

        ## diffusion model
        'horizon': 256,
        'n_diffusion_steps': 256,
        'normalizer': 'LimitsNormalizer',

        ## serialization
        'vis_freq': 10,
        'logbase': 'logs',
        'prefix': 'plans_finetune/release',
        'exp_name': watch(plan_args_to_watch),
        'suffix': '0',
    
        'conditional': True,
        'replan': False,

        ## loading
        'diffusion_loadpath': 'f:diffusion_finetune/H{horizon}_T{n_diffusion_steps}_Eevolution',
        'diffusion_epoch': 'latest',
    },

}

#------------------------ overrides ------------------------#

'''
    maze2d maze episode steps:
        umaze: 150
        medium: 250
        large: 600
'''

maze2d_umaze_v1 = {
    'diffusion': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
    'plan': {
        'horizon': 128,
        'n_diffusion_steps': 64,
    },
}

maze2d_large_v1 = {
    'diffusion': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
    'plan': {
        'horizon': 384,
        'n_diffusion_steps': 256,
    },
}
