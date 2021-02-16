import sys
import gym
import numpy as np
import os
import datetime

from gym import spaces

import baselines.run as blr

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from sacred import Experiment
from sacred.observers import FileStorageObserver

class bl_arg_class():
    pass

alg_global = 'a2c'
log_path_global = os.path.join('/mnt/ma_neural_map/nm_results', alg_global +  datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S-%f"))

ex = Experiment(save_git_info=False)
ex.observers.append(FileStorageObserver(log_path_global))


# BEGIN PARAMETER SECTION --> @ex.config() #########################################################
@ex.config
def nm_config():
    # baseline's general parameters
    env = 'neural_map_envs:two_indicators_goals_dimensions-v0'
    env_type = None
    seed = int(2511988)
    alg = alg_global
    num_timesteps = 1e3
    network = None
    gamestate = None
    num_env = 1
    reward_scale = 1.0
    save_path = None
    save_video_interval = int(0)
    save_video_length = int(200)
    play = False

    log_path = log_path_global

    # alg's parameters
    alg_args = {}

    # ppo2's parameters
    if alg == 'ppo2':
        print('\nPPO2 algorithm will be used!\n')
        # default = None
        alg_args['eval_env'] = None
        # number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv); default = 2048
        alg_args['nsteps'] = 2048
        # policy entropy coefficient in the optimization objective; default = 0.0
        alg_args['ent_coef'] = 0.0
        # learning rate, constant or a schedule function [0,1] -> R+ where 1 is beginning of the training
        # and 0 is the end of the training; default = 3e-4
        alg_args['lr'] = 1e-4
        # value function loss coefficient in the optimization objective; default = 0.5
        alg_args['vf_coef'] = 0.5
        # gradient norm clipping coefficient; default = 0.5
        alg_args['max_grad_norm'] = 100
        # discounting factor; default = 0.99
        alg_args['gamma'] = 0.99
        # advantage estimation discounting factor (lambda in the paper); default = 0.95
        alg_args['lam'] = 0.95
        # number of timesteps between logging events; default = 10
        alg_args['log_interval'] = 10
        # number of training minibatches per update. For recurrent policies, should be smaller
        # or equal than number of environments run in parallel; default = 4
        alg_args['nminibatches'] = 4
        # number of training epochs per update; default = 4
        alg_args['noptepochs'] = 4
        # clipping range, constant or schedule function [0,1] -> R+ where 1 is beginning of the training
        # and 0 is the end of the training; default = 0.2
        alg_args['cliprange'] = 0.2
        # number of timesteps between saving events; default = 0
        alg_args['save_interval'] = 0
        # path to load the model from; default = None
        alg_args['load_path'] = None
        # default = None
        alg_args['model_fn'] = None
        # default = None
        alg_args['update_fn'] = None
        # default = None
        alg_args['init_fn'] = None
        # default = 1
        alg_args['mpi_rank_weight'] = 1
        # default = None
        alg_args['comm'] = None

    # a2c's parameters
    elif alg == 'a2c':
        print('\nA2C algorithm will be used!\n')
        # number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv); default = 5
        alg_args['nsteps'] = 5
        # coefficient in front of value function loss in the total loss function; default = 0.5
        alg_args['vf_coef'] = 0.5
        # coeffictiant in front of the policy entropy in the total loss function; default = 0.01
        alg_args['ent_coef'] = 0.01
        # gradient is clipped to have global L2 norm no more than this value; default = 0.5
        alg_args['max_grad_norm'] = 100
        # learning rate for RMSProp (current implementation has RMSProp hardcoded in); default = 7e-4
        alg_args['lr'] = 5e-3
        # schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction
        # of the training progress as input and returns fraction of the learning rate as output; default = 'linear'
        alg_args['lrschedule'] = 'linear'
        # RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update; default = 1e-5
        alg_args['epsilon'] = 1e-5
        # RMSProp decay parameter; default = 0.99
        alg_args['alpha'] = 0.99
        # reward discounting parameter; default = 0.99
        alg_args['gamma'] = 0.99
        # specifies how frequently the logs are printed out; default = 100
        alg_args['log_interval'] = 100
        # path to load the model from; default = None
        alg_args['load_path'] = None

    # deepq's parameters
    elif alg == 'deepq':
        print('\nDeepQ algorithm will be used!\n')
        # learning rate; default = 5e-4
        alg_args['lr'] = 1e-4
        # size of the replay buffer; default = 50000
        alg_args['buffer_size']=50000
        # fraction of entire training period over which the exploration rate is annealed
        alg_args['exploration_fraction']=0.1
        # final value of random action probability; default = 0.2
        alg_args['exploration_final_eps']=0.02
        # update the model every 'train_freq' steps; default = 1
        alg_args['train_freq']=1
        # batch size for training / replay buffer; default = 32
        alg_args['batch_size']=16
        # how often to printout training progress; None -> no printout; default = 100
        alg_args['print_freq']=100
        # how often to save the model;
        alg_args['checkpoint_freq']=10000
        # default = None
        alg_args['checkpoint_path']=None
        # training starts after 'learning_starts' transitions are collected; default = 1000
        alg_args['learning_starts']=1000
        # discount factor; default = 1.0
        alg_args['gamma'] = 0.99
        # update the target network every 'target_network_update_freq' steps; default = 500
        alg_args['target_network_update_freq']=500
        # if True prioritized replay buffer will be used; default = False
        alg_args['prioritized_replay']=False
        # default = 0.6
        alg_args['prioritized_replay_alpha'] = 0.6
        # default = 0.4
        alg_args['prioritized_replay_beta0'] = 0.4
        # default = None
        alg_args['prioritized_replay_beta_iters'] = None
        # default = 1e-6
        alg_args['prioritized_replay_eps'] = 1e-6
        # default = False
        alg_args['param_noise'] = False
        # default = None
        alg_args['callback'] = None
        # path to load the model from; default = None
        alg_args['load_path'] = None

    # neural map's parameters
    if env.split(':')[0] == 'neural_map_envs':
        alg_args['nm_customization_args'] = {'use_nm_customization':True,
                                             'log_model_parameters':True,
                                             'log_path':log_path_global,
                                             'optimizer':'RMSProp'}

    if network == 'neural_map':
        # neural map's dimensions as a list in order horizontal dim, vertical dim, c_dim
        alg_args['nm_dims'] = [5, 5, 32]
        # global read's args; list of dicts / ints where every dict / int contains one layer's parameter(s)
        # has to contain at least 1 dict with conv layer's parameters
        # the last fc layer doesn't have to be specified and always has c_dim neurons
        alg_args['gr_args'] = [{'nf':8, 'rf':3, 'stride':1, 'pad':[[0,0], [1,1], [1,1], [0,0]]},
                               {'nf':8, 'rf':3, 'stride':2, 'pad':[[0,0], [0,0], [0,0], [0,0]]},
                               256]
        # local write's args; list that contains the number of neurons in the fc layers
        # the last fc layer doesn't have to be specified and always has c_dim neurons
        alg_args['lw_args'] = [256]
        # final nn's args; list that contains the number of neurons in the fc layers
        # the last fc layer doesn't have to be specified and always has nactions neurons
        alg_args['fnn_args'] = [256]
        # number of actions
        alg_args['nactions'] = 3

# END PARAMETER SECTION --> @ex.config() #########################################################



# BEGIN RUN SECTION --> @ex.automain #############################################################
@ex.automain
def nm_main(env,
            env_type,
            seed,
            alg,
            num_timesteps,
            network,
            gamestate,
            num_env,
            reward_scale,
            save_path,
            save_video_interval,
            save_video_length,
            play,
            log_path,
            alg_args):

    bl_args = bl_arg_class()
    bl_args.env = env
    bl_args.env_type = env_type
    bl_args.seed = seed
    bl_args.alg = alg
    bl_args.num_timesteps = num_timesteps
    bl_args.network = network
    bl_args.gamestate = gamestate
    bl_args.num_env = num_env
    bl_args.reward_scale = reward_scale
    bl_args.save_path = save_path
    bl_args.save_video_interval = save_video_interval
    bl_args.save_video_length = save_video_length
    bl_args.log_path = log_path
    bl_args.play = play

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        blr.configure_logger(bl_args.log_path)
    else:
        rank = MPI.COMM_WORLD.Get_rank()
        blr.configure_logger(bl_args.log_path, format_strs=[])

    model, env = blr.train(bl_args, alg_args)

    #if save_path is not None and rank == 0:
    #    save_path = os.path.expanduser(save_path)
    #    model.save(save_path)

    env.close()

    print('\nDie allerallerallerletzte Zeile...\n')
# END RUN SECTION --> @ex.automain #############################################################
