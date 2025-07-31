import socket
 
from diffuser.utils import watch
# this config file is used to train a cost guide model for diffuser in the maze2d environment. 
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

        ## policy
        'policy' : 'guides.guided_policies.GuidedPolicy',
        'verbose': True,
        'n_guide_steps' :2,
        'scale':10000, 
        'sample_fn': 'utils.n_step_guided_p_sample',
        #'exp': 'guide_s11_g79' ,
        't_stopgrad' :2,
        'scale_grad_by_std': True,
        'return_diffusion':False,

        ## dataset
        'loader': 'datasets.GoalDataset',
        'termination_penalty': None,
        'normalizer': 'LimitsNormalizer',
        'preprocess_fns': ['maze2d_set_terminals'],
        'clip_denoised': True,
        'use_padding': False,
        'max_path_length': 40000,

        ## serialization
        'logbase': 'logs',
        'prefix': 'diffusion_cost/',
        'exp_name': watch(diffusion_args_to_watch),

        ## training
        'trainer': 'utils.CostTrainer',
        'conditional': True, # whether to randomly set the goal to train.
        #'n_steps_per_epoch': 800,
        'loss_type': 'l2',
        'n_train_episodes': 1e2,
        'sample_batch_size': 10,
        'batch_size': 32,
        'learning_rate': 1e-4,
        #'gradient_accumulate_every': 2,
        #'ema_decay': 0.995,
        'save_freq': 5,
        #'sample_freq': 10,
        'test_freq' : 5,
        'n_test_samples': 1,
        'n_saves': 20,
        'epsilon': 1.0, # the distance need to replan
        'update_guide_freq':5,
        'save_parallel': False,
        #'n_reference': 50,
        #'n_samples': 10,
        'bucket': None,
        'device': 'cuda',
        ## loading
        #'finetune': True,
        'loadpath': '/home/junda/diffuser/logs/maze2d-large-v1/diffusion/H384_T256/state_1960000.pt',
    },

    'cost': {
        'model': 'models.MultiLinearLayer',
        'hidden_dim': 64,
        'n_layers': 2,
        'neighbour_num': 4, # 4 neighbours, 8 neighbours, 9 include the center

        'device': 'cuda',
        'cost':'guides.guides.CostGuide_maze2d',

        'buffer': 'datasets.buffer.ReplayBufferSAS',
        'buffer_size': 4000,

        
        ## loading；
        'loadpath': None,
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
