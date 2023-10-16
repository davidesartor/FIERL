#!/usr/bin/env python

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import random 
import ast
from copy import deepcopy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")

from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
from envs.faultobserver.observerlogger import *
from scripts.test_utils import * 
from scripts.pid import PID
from envs.env_utils import sample_uniformly_from_ball

def run_comparison(env, get_action, 
                   setpoints:list, kp:list, ki:list=None, kd:list=None, Ts:float=None,
                   max_ep_len:int = None, num_episodes:int = 100, render:bool = True, 
                   faults_list:list = None, fault_change_step:int = None, 
                   faults_number:int = None, min_steps:int = None, max_steps:int = None, adaptive_len:bool = False,
                   save:bool = True, save_file_name:str = None, save_format:str = 'pdf',
                   overlapped_plot:bool = True):
    """
    Run num_episodes of 2 environments, one with the trained policy loaded in get_action and the other with a noisy PID controller
    under the same initial condition. Compare the results. 
    
    Args: 
        env: environment 
        get_action: function that returns the trained policy action given an observation

        setpoints (list[np.ndarray]): control targets
        kp (list[float]): proportional gains 
        ki (list[float]): integral gains
        kd (list[float]): derivative gains
        Ts (float): sampling time

        max_ep_len (int): length of an episode
        num_episodes (int): number of episodes
        render (bool): if True, render the environment 

        faults_list (list): list of faults. If None, faults are randomly sampled 
        fault_change_step (int): step at which to change the fault. If None, steps to change are randomly sampled 
        faults_number (int): number of fault to change, not needed if adaptive_len is True 
        min_steps (int): minimum number of steps between fault changes
        max_steps (int): maximum number of steps between fault changes
        adaptive_len (bool): if True, expand the length of the episode to ensure the number of randomly generated faults

        save (bool): if True, save the plots
        save_file_name (str): name of the file where to save the plots
        save_format (str): format of the file to save the plots

        overlapped_plot (bool): if True, plot the environments in the same figure
    """

    print(env.system.state_noise_std)
    env.is_train = False
    if max_ep_len > len(env.ref): 
        env.ref = adjust_list_length(env.ref, max_ep_len)

    env_RL = deepcopy(env)
    env_PID = deepcopy(env)

    logger_RL = EpochLogger()
    logger_PID = EpochLogger()

    returns_RL, c_returns_RL = [], []
    returns_PID, c_returns_PID = [], []

    normalized_returns_RL, normalized_c_returns_RL = [], []
    normalized_returns_PID, normalized_c_returns_PID = [], []

    ki = [ki_i * Ts for ki_i in ki] if ki is not None else np.zeros_like(kp)
    kd = [kd_i / Ts for kd_i in kd] if kd is not None else np.zeros_like(kp)

    for n in range(num_episodes): 
        true_ep_len = max_ep_len
        pid = PID(kp = kp, ki = ki, kd = kd, input_dim = env.system.input_dim, output_dim = env.system.output_dim)

        if fault_change_step is None: 
            # random fault changes
            if faults_number is not None:  
                if min_steps is not None and max_steps is not None: 
                    changesteps = ordered_constrained_fixedlen_list(max_value=true_ep_len, min_diff=min_steps, max_diff=max_steps, max_len=faults_number - 1)
                else:
                    changesteps = sorted(random.sample(range(1, true_ep_len), faults_number-1))
            else: 
                if min_steps is not None and max_steps is not None: 
                    changesteps = ordered_constrained_list(true_ep_len, min_steps, max_steps, adaptive_len)
                    if adaptive_len: 
                        true_ep_len = changesteps[-1]
                        setpoints = adjust_list_length(setpoints, true_ep_len)
                        env_RL.ref = adjust_list_length(env_RL.ref, true_ep_len)
                        env_PID.ref = adjust_list_length(env_PID.ref, true_ep_len)
                else: 
                    raise ValueError("If fault_change_Step is none, you must provide either faults_number (int) or (min_steps and max_steps)")
        else: 
            changesteps = [fault_change_step * (i + 1) for i in range(int(max_ep_len/fault_change_step) -1 )]
    
        change_step_index = 0
       
        if faults_list is None: 
            # random faults
            fault = np.array([random.uniform(0, 1) for _ in range(env_RL.system.input_dim)])
            faults_string = np.array2string(fault, formatter={'float_kind': '{:.3f}'.format}) + '_'
        else: 
            fault_index = 0
            fault = np.array(faults_list[fault_index])
            faults_string = str(faults_list[fault_index]) + '_'
        
        # reset the environment 
        o_RL = env_RL.reset(initial_fault = fault, obs_logger = ObserverLogger())
        r_RL, ep_ret_RL, ep_cost_RL, ep_len= 0, 0, 0, 0

        o_PID = env_PID.reset(initial_fault = fault, obs_logger = ObserverLogger())
        r_PID, ep_ret_PID, ep_cost_PID, ep_len= 0, 0, 0, 0

        for t in range(true_ep_len): 
            a_RL = get_action(o_RL)
            o_RL, r_RL, d_RL, info_RL = env_RL.step(a_RL)

            a_PID = pid.step(setpoints[t], env_PID.dict_s['system_output']).flatten()
            o_PID, r_PID, d_PID, info_PID = env_PID.step(a_PID)

            ep_ret_RL += r_RL
            ep_cost_RL += info_RL['cost']
            ep_ret_PID += r_PID
            ep_cost_PID += info_PID['cost']
            ep_len += 1

            if change_step_index < len(changesteps) and t == changesteps[change_step_index]: 
                if faults_list is None: 
                    fault = np.array([random.uniform(0, 1) for _ in range(env_RL.system.input_dim)])
                    env_RL.set_fault(fault)
                    env_PID.set_fault(fault) 
                    faults_string += np.array2string(fault, formatter={'float_kind': '{:.3f}'.format}) + '_'
                else: 
                    fault_index += 1
                    env_RL.set_fault(np.array(faults_list[fault_index]))
                    env_PID.set_fault(np.array(faults_list[fault_index]))
                    faults_string += str(faults_list[fault_index]) + '_'             
                change_step_index += 1
        
            logger_RL.store(EpRet = ep_ret_RL, EpCost = ep_cost_RL, EpLen = ep_len)
            logger_PID.store(EpRet = ep_ret_PID, EpCost = ep_cost_PID, EpLen = ep_len)

        if num_episodes == 1: 
            print('RL: Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret_RL, ep_cost_RL, ep_len))
            print('P : Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret_PID, ep_cost_PID, ep_len))

        returns_RL.append(ep_ret_RL)
        c_returns_RL.append(ep_cost_RL)
        normalized_returns_RL.append(ep_ret_RL/ep_len)
        normalized_c_returns_RL.append(ep_cost_RL/ep_len)

        returns_PID.append(ep_ret_PID)
        c_returns_PID.append(ep_cost_PID)
        normalized_returns_PID.append(ep_ret_PID/ep_len)
        normalized_c_returns_PID.append(ep_cost_PID/ep_len)

        if render: 
            if save_file_name is None: 
                if fault_change_step is not None: 
                    save_file_name = 'G' + faults_string + 'Episodes' + str(num_episodes) + '_steps' +  str(max_ep_len) + '_change' + str(fault_change_step) + '_' + str(n)
                else: 
                    save_file_name = 'G' + faults_string + 'Episodes' + str(num_episodes) + '_steps' +  str(max_ep_len) + '_' + str(n)
        
            if not overlapped_plot:
                env_RL.render(save = save, save_name = 'RL_' + save_file_name + '.' + str(save_format), title_prefix = 'RL ')
                env_PID.render(save = save, save_name = 'PID_' +  save_file_name + '.' + str(save_format), title_prefix = 'PID ')
            else: 
                plot_comparison(env_RL.obs_logger, env_PID.obs_logger, data_type='state', labels=['RL', 'P'], save=True, save_name = save_file_name + '.' + str(save_format))
                plot_comparison(env_RL.obs_logger, env_PID.obs_logger, data_type='fault', labels=['RL', 'P'], save=True, save_name = save_file_name + '.' + str(save_format))
                plot_comparison(env_RL.obs_logger, env_PID.obs_logger, data_type='output', labels=['RL', 'P'], save=True, save_name = save_file_name + '.' + str(save_format))
                plot_comparison(env_RL.obs_logger, env_PID.obs_logger, data_type='control_input', labels=['RL', 'P'], save=True, save_name = save_file_name + '.' + str(save_format))

            # compute and plot the tracking error for RL agent
            output_log_array_RL = np.array(env_RL.obs_logger.output_log)
            ref_array_RL = adjust_list_length(env_RL.ref, len(env_RL.obs_logger.output_log))
            track_error_norm_RL = np.linalg.norm(output_log_array_RL - ref_array_RL, axis = 1)

            plt.figure(figsize = (6, 3.25))
            plt.step(env_RL.obs_logger.time_log, track_error_norm_RL, '-', linewidth = 1.5, label = 'Tracking error')
            plt.plot(env_RL.obs_logger.time_log, [env_RL.tracking_threshold]* len(env_RL.obs_logger.time_log), 'r--', label = 'threshold')
            plt.xlabel('time (s)')
            plt.ylabel('tracking error')
            plt.legend(loc = 'lower right')
            plt.xlim([0, env_RL.obs_logger.time_log[-1]])
            plt.title('RL Tracking Error')
            if save: 
                save_file_name_trackerr = 'RL Tracking error' if save_file_name is None else 'RL Tracking_error_' + save_file_name + '.' + str(save_format)
                dpi = 400 if save_file_name_trackerr.endswith(".png") else None
                plt.savefig(save_file_name_trackerr, dpi=dpi, bbox_inches='tight')
            
            # compute and plot the tracking error for PID agent
            output_log_array_PID = np.array(env_PID.obs_logger.output_log)
            ref_array_PID = adjust_list_length(env_PID.ref, len(env_PID.obs_logger.output_log))
            track_error_norm_PID = np.linalg.norm(output_log_array_PID - ref_array_PID, axis = 1)

            plt.figure(figsize = (6, 3.25))
            plt.step(env_PID.obs_logger.time_log, track_error_norm_PID, '-', linewidth = 1.5, label = 'Tracking error')
            plt.plot(env_PID.obs_logger.time_log, [env_PID.tracking_threshold]* len(env_PID.obs_logger.time_log), 'r--', label = 'threshold')
            plt.xlabel('time (s)')
            plt.ylabel('tracking error')
            plt.legend(loc = 'lower right')
            plt.xlim([0, env_PID.obs_logger.time_log[-1]])
            plt.title('P Tracking Error')
            if save: 
                save_file_name_trackerr = 'P Tracking error' if save_file_name is None else 'P Tracking_error_' + save_file_name + '.' + str(save_format)
                dpi = 400 if save_file_name_trackerr.endswith(".png") else None
                plt.savefig(save_file_name_trackerr, dpi=dpi, bbox_inches='tight')
                   
            
        if num_episodes == 1 and save: 
            
            if fault_change_step is not None: 
                save_text_name_RL = 'RL_' + save_file_name + '.txt' if save_file_name is not None else 'RL_' + faults_string + '_' + str(num_episodes) + 'Ep_' + str(max_ep_len) + '_change' + str(fault_change_step) + '_' + str(n) + '.txt'
                save_text_name_PID = 'P_' + save_file_name + '.txt' if save_file_name is not None else 'P_' + faults_string + '_' + str(num_episodes) + 'Ep_' + str(max_ep_len) + '_change' + str(fault_change_step) + '_' + str(n) + '.txt'
            else: 
                save_text_name_RL = 'RL_' + save_file_name + '.txt' if save_file_name is not None else 'RL_' + faults_string + '_' + str(num_episodes) + 'Ep_' + str(max_ep_len) + '_' + str(n) + '.txt'
                save_text_name_PID = 'P_' + save_file_name + '.txt' if save_file_name is not None else 'P_' + faults_string + '_' + str(num_episodes) + 'Ep_' + str(max_ep_len) + '_' + str(n) + '.txt'

            with open(save_text_name_RL, 'a') as f: 
                f.write("Episode \t{}\n".format(n))                    
                f.write("EpRet \t{}\n".format(ep_ret_RL))
                f.write("EpCost \t{}\n".format(ep_cost_RL))
                f.write("EpLen \t{}\n".format(ep_len))
                f.write("Faults \t{}\n".format(faults_string))
                f.write("Final_estimate \t{}\n".format(env_RL.dict_s["fault_estimate"]))
                f.write("\n")
            
            with open(save_text_name_PID, 'a') as f: 
                f.write("Episode \t{}\n".format(n))                    
                f.write("EpRet \t{}\n".format(ep_ret_PID))
                f.write("EpCost \t{}\n".format(ep_cost_PID))
                f.write("EpLen \t{}\n".format(ep_len))
                f.write("Faults \t{}\n".format(faults_string))
                f.write("Final_estimate \t{}\n".format(env_PID.dict_s["fault_estimate"]))
                f.write("\n")

        logger_RL.log_tabular('EpRet', with_min_and_max=True)
        logger_RL.log_tabular('EpCost', with_min_and_max=True)
        logger_RL.log_tabular('EpLen', average_only=True)

        logger_PID.log_tabular('EpRet', with_min_and_max=True)
        logger_PID.log_tabular('EpCost', with_min_and_max=True)
        logger_PID.log_tabular('EpLen', average_only=True)

    
        print("RL:")
        print_statistics('Returns', returns_RL)
        print_statistics('Constraint violations', c_returns_RL)
        print_statistics('Normalized returns', normalized_returns_RL)
        print_statistics('Normalized constraint violations', normalized_c_returns_RL)

        print("\nP:")
        print_statistics('Returns', returns_PID)
        print_statistics('Constraint violations', c_returns_PID)
        print_statistics('Normalized returns', normalized_returns_PID)
        print_statistics('Normalized constraint violations', normalized_c_returns_PID)

        if save:    
            save_text_name_RL = 'RL_' +  save_file_name + '.txt' if save_file_name is not None else 'RL Statistics_' + str(num_episodes) + 'Ep.txt'
            save_statistics_to_file(save_text_name_RL, 'Returns', returns_RL)
            save_statistics_to_file(save_text_name_RL, 'Constraint violations', c_returns_RL)
            save_statistics_to_file(save_text_name_RL, 'Normalized Returns', normalized_returns_RL)
            save_statistics_to_file(save_text_name_RL, 'Constraint violations', normalized_c_returns_RL)

            save_text_name_PID = 'P_' + save_file_name + '.txt' if save_file_name is not None else 'P Statistics_' + str(num_episodes) + 'Ep.txt'
            save_statistics_to_file(save_text_name_PID, 'Returns', returns_PID)
            save_statistics_to_file(save_text_name_PID, 'Constraint violations', c_returns_PID)
            save_statistics_to_file(save_text_name_RL, 'Normalized Returns', normalized_returns_PID)
            save_statistics_to_file(save_text_name_RL, 'Constraint violations', normalized_c_returns_PID)

        if render:
            plt.show(block=False)
            input("Hit [enter] to end.")
            plt.close('all')

if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=120)
    parser.add_argument('--episodes', '-n', type=int, default=1)
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--norender', '-norend', action='store_true')   

    parser.add_argument('--faults_list', '-fl', type=str, default = None)
    parser.add_argument('--fault_change_step', '-fcs', type=int, default=None)
    parser.add_argument('--faults_number', '-fn', type=int, default=None)
    required_group = parser.add_argument_group('Min and max steps')
    required_group.add_argument('--min_steps', '-mins', type=int, default=None)
    required_group.add_argument('--max_steps', '-maxs', type=int, default=None)

    parser.add_argument('--kp', '-kp', type=float, nargs='+', default=[0.016,0.016])
    parser.add_argument('--kd', '-kd', type=float, nargs='+', default=None)
    parser.add_argument('--ki', '-ki', type=float, nargs='+', default=None)
    parser.add_argument('--Ts', '-Ts', type=float, default=0.1)

    parser.add_argument('--save', '-s', action='store_false')
    parser.add_argument('--save_file_name', '-sfnm', type=str, default=None)
    parser.add_argument('--save_format', '-sfmt', type=str, default='pdf')

    parser.add_argument('--adaptive_len', '-al', action='store_true')
    parser.add_argument('--overlapped_plots', '-op', action='store_true')

    args = parser.parse_args()

"""
Args: 
    
    fpath (str):            path to the folder where the model is saved
    len (int):              length of an episode
    episodes (int):         number of episodes
    itr (int):              number of the iteration of the model to load
    deterministic (bool):   if True, use the deterministic version of the policy
    norender (bool):        if present, do not render the environment 

    faults_list (list, optional):       list of faults. Example of correct form --faults [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                                        if not given, then random faults are generated. 

    fault_change_step (int, optional):  number of steps between fault changes. 
                                        If not given, then the change steps are randomly sampled. 
    
    faults_number (int, optional):      number of faults to change
    min_steps (int, optional):          minimum number of steps between fault changes. 
                                        Needed only if fault_change_step is None. 
    max_steps (int, optional):          maximum number of steps between fault changes. 
                                        Needed only if fault_change_step is None. 
    
    kp (list[float]): proportional gains
    ki (list[float]): integral gains
    kd (list[float]): derivative gains
    Ts (float): sampling time
 
    save (bool): if present, the plots are saved
    save_file_name (str, optional): name of the file where to save the plot
    save_format (str, optional): format of the file to save the plot 

    adaptive_len (bool):    if present, the length of the episode is expanded to ensure the number of randomly generated faults  
    overlapped_plots (bool): if present it shows the same component of each signal (state, fault, output, control)
                             for RL and P policy in the same plot.   
"""

env, get_action, sess = load_policy(args.fpath, args.itr if args.itr >=0 else 'last', args.deterministic)

if args.fault_change_step is None and (args.faults_number is None and (args.min_steps is None and args.max_steps is None) and args.faults_list is None): 
    raise ValueError("At least one between min_steps (int) and max_steps (int) must be given if fault_change_step is None and faults_list is None")

# parsed fault list
if args.faults_list is not None: 
    try: 
        parsed_faults = ast.literal_eval(args.faults_list)
        if not isinstance(parsed_faults, list) or any(not isinstance(sublist, list) for sublist in parsed_faults):
            raise ValueError("Invalid format for 'faults_list'")
    except:
        print("\033[91mError:\033[0m Invalid format for 'faults'. Correct example: \"[[1, 2, 3], [4, 5, 6], [7, 8, 9]]\"")
        exit(1)
else: 
    parsed_faults = None

setpoints = adjust_list_length(env.ref, args.len)
setpoints = [s_i + np.random.uniform(-env.tracking_threshold*np.sqrt(3), env.tracking_threshold*np.sqrt(3), size=np.shape(s_i)) for s_i in setpoints] 
# for i in range(len(setpoints)):
#     setpoints[i] = sample_uniformly_from_ball(setpoints[i], env.tracking_threshold, env.system.output_dim)   

if args.fault_change_step is None and (args.faults_number is None and (args.min_steps is None and args.max_steps is None)):
    args.faults_number = len(parsed_faults) 

run_comparison(env=env, get_action=get_action, 
               setpoints = setpoints, kp = args.kp, ki = args.ki, kd = args.kd, Ts = args.Ts,
               max_ep_len = args.len, num_episodes = args.episodes, render = not(args.norender), 
               faults_list = parsed_faults if args.faults_list is not None else None, fault_change_step = args.fault_change_step,
               faults_number = args.faults_number, min_steps = args.min_steps, max_steps = args.max_steps, 
               adaptive_len = args.adaptive_len, 
               save = args.save, save_file_name = args.save_file_name, save_format = args.save_format, 
               overlapped_plot=args.overlapped_plots)



