import socket
 
from diffuser.utils import watch
# this config file is used to train a cost guide model for diffuser in the maze2d environment.
# the robust version means interact with no guide, then train cost to  convergence, finally test with different guide scale.
 
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
        'renderer': 'utils.Maze2dRendererBlock',

        ## policy
        'policy' : 'guides.guided_policies.GuidedPolicy',
        'verbose': True,
        'n_guide_steps' :2,
        'scale':0, 
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

        ## trainer
        'trainer': 'utils.CostRobustTrainer',
        'exp' : 'cost_robust_3',
        'conditional': True, # whether to randomly set the goal to train.
        #'n_steps_per_epoch': 800,
        'loss_type': 'l2',
        #'n_train_episodes': 1e2,
        ## collect
        'n_collect_episodes': 25,
        'vis_collect_freq' : 5,
        'sample_batch_size': 10,

        ## train_cost
        'batch_size': 32,
        'learning_rate': 1e-4,
        'n_train_epochs': 5,
        #'gradient_accumulate_every': 2,
        #'ema_decay': 0.995,
        'save_freq': 1,  #epoch
        #'sample_freq': 10,

        # test
        'n_test_samples': 20,
        'vis_test_freq':5,
        
        'epsilon': 1.0, # the distance need to replan
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
        'n_layers': 4,
        'neighbour_num': 8, # 4 neighbours, 8 neighbours, 9 include the center
        #'scales' : [0, 10, 100, 1000, 10000, 100000, 1000000], # different guide scale for test
        'scales' : [0, 10, 30, 50],
        'device': 'cuda',
        'cost':'guides.guides.CostGuide_maze2d',

        'buffer': 'datasets.buffer.ReplayBufferSAS',
        'buffer_size': 20000, # 25 * 800

        
        ## loading；
        'loadpath': '/home/junda/diffuser/logs/maze2d-large-v1/diffusion_cost/H384_T256_Ecost_robust_2/cost_model_4.pt',
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
