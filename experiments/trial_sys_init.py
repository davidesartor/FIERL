from __future__ import annotations

import os
import sys
import time
import random
from collections import deque
from typing import Callable
from distutils.util import strtobool
import argparse
import numpy as np
import gymnasium as gym
import torch

import wandb

# sys.path.append('/root/FIERL')

from environment.env import Environment
from environment.systems.trial_system import ProvaSystem
from environment.faultobserver.faultobserver import FaultObserver, GaussianEstimate
from environment.utils import InitialConditionSampler, FaultSampler


from safepo.single_agent.cpo_initialization import train_cpo 


training_config_default = {
    'hidden_sizes': [512, 512, 512], 
    'gamma': 0.99, 
    'target_kl': 0.01, 
    'batch_size': 128, 
    'learning_iters': 10, 
    'max_grad_norm': 40, 
    'steps_per_epoch': 4000, 
    'total_steps': 4000*1000,
    'log_std_param': True, # if False the std is a layer, if True the std is a learnable parameter
    'critic_lr':1e-3,
    'cost_limit': 3.0, 
    'action_offset': 0,
}

cfg_env_default = {
    'max_ep_len': 20, 
    'tracking_threshold': 0.1, 
    'reference' : [np.array([[0.5]])] * 20,

    # information on system
    'min_action': -10, #np.array([-10, -10, 0.5]), #
    'max_action': 10, #np.array([10, 10, 0.5]), #0.02
    'state_noise_std': 1e-5, 
    'output_noise_std':1e-5, 
    
    # information on faultObserver
    'fault_evol_cov': 1e-8,

    # information on sampler of initial condition
    'initial_condition_fnc': InitialConditionSampler,
    'ic_type':'ncube',  
    'half_side':1e-3,
    'center': 1, #np.array([1, 1, 1]), # metterlo più generale: se int, allora lo inizializza a dim state_dim, cambiare anche su cond iniziali
    
    # information on fault generator
    'fault_generator_fnc': SpecialFaultSampler,
    'fault_type': 'uniform', 
    'a': 0, 
    'b': 1,
}


def main(args, env_config = None, training_config = None): 
    env_config = env_config if env_config is not None else cfg_env_default
    training_config = training_config if training_config is not None else training_config_default

    # add env and training config to args 
    args.config = training_config
    args.env_cfg = env_config

    # define environment 

    system = ProvaSystem(state_noise_std=env_config['state_noise_std'], 
                         output_noise_std=env_config['output_noise_std'], 
                         min_input=env_config['min_action'], 
                         max_input=env_config['max_action'])  
    env_config['center'] = np.ones(system.state_dim) * env_config['center']  
    fault_observer = FaultObserver(state_dim=system.state_dim,
                                   input_dim=system.input_dim,
                                   fault_evol_cov=env_config['fault_evol_cov'])
    ic_sampler = InitialConditionSampler(ic_type = env_config['ic_type'],
                                         half_side=env_config['half_side'],
                                         center=env_config['center'],
                                         n_samples=1)
    fault_sampler = FaultSampler(fault_type = env_config['fault_type'],
                                 a = env_config['a'],
                                 b = env_config['b'],
                                 size = (system.input_dim, ))      
    env = Environment(system = system,
                      fault_observer = fault_observer,
                      reference = env_config['reference'],
                      track_threshold= env_config['tracking_threshold'],
                      initial_condition_fnc=ic_sampler,
                      fault_generator_fnc=fault_sampler,
                      max_ep_len=args.max_ep_len)
    
    train_cpo(env, args)


if __name__ == '__main__':
    custom_params = [
        {'name': '--seed', 'type': int, 'default':0, 'help':'random seed'}, 
        {'name': '--use-eval', 'action':'store_true', 'help': 'Use evaluation environment for testing'},
        {'name': '--num-envs', 'type': int, 'default':1, 'help':'Number of environments to run in parallel'},
        {'name': '--task', 'type': str, 'default':'', 'help':'Task name'},
        {'name': '--experiment', 'type': str, 'default': 'ProvaSystem', 'help':'Experiment name'},
        {'name': '--log-dir', 'type': str, 'default': 'runs', 'help':'Directory to save logs'},
        {'name': '--device', 'type': str, 'default': 'cpu', 'help':'Device to run on'},
        {'name': '--device-id', 'type': int, 'default': 0, 'help':'Device id to run on'},
        {'name': '--write-terminal', 'action': 'store_false', 'help': 'Toggles terminal logging'},
        {'name': '--headless', 'action': 'store_true', 'help': 'Toggles headless mode'},
        {'name': '--total-steps', 'type': int, 'default': 4000*500, 'help':'Total timesteps of the experiment'},
        {'name': '--steps-per-epoch', 'type': int, 'default': 4000, 'help':'Number of steps to run in each environment per policy rollout'},
        {'name': '--max_ep_len', 'type': int, 'default':20, 'help':'Maximum length of the episodes'}, 
        {'name': '--cost-limit', 'type': float, 'default': 0.0, 'help':'Cost limit for the environment'},
        {'name': '--lagrangian-multiplier-init', 'type': float, 'default': 0.001, 'help': 'initial value of lagrangian multiplier'},
        {'name': '--lagrangian-multiplier-lr', 'type': float, 'default': 0.035, 'help': 'learning rate of lagrangian multiplier'},
        {'name': '--use-wandb', 'action': 'store_true', 'help': 'Toggles wandb logging'},
        {'name': '--wandb-project', 'type': str, 'default': 'FIERL-sys prova', 'help': 'Wandb project name'},
        {'name': '--wandb-save', 'action': 'store_true', 'help': 'Toggles wandb saving'},
        ]
    
    parser = argparse.ArgumentParser(description='CPO_prova')
    for param in custom_params:
        param_name = param.pop('name')
        parser.add_argument(param_name, **param)
    
    args = parser.parse_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.experiment, args.task, algo, relpath)

    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args)
    else:
        main(args)