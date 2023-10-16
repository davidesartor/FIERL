#!/usr/bin/env python

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import random 
import ast


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore")

from safe_rl.utils.load_utils import load_policy
from safe_rl.utils.logx import EpochLogger
from envs.faultobserver.observerlogger import ObserverLogger
from scripts.test_utils import * 


def run_policy(env, get_action, 
               max_ep_len:int = None, num_episodes:int = 100, render:bool = True, 
               faults_list:list = None, fault_change_step:int = None, 
               faults_number:int = None, min_steps:int = None, max_steps:int = None, adaptive_len:bool = False,
               save:bool = True, save_file_name:str = None, save_format:str = 'pdf'):
    """
    Run num_episodes of the environment with the policy loaded in get_action
    
    Args: 
        env: environment 
        get_action: function that returns the trained policy action given an observation
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
    """
    env.is_train = False 
    
    if max_ep_len > len(env.ref): 
        env.ref = adjust_list_length(env.ref, max_ep_len)

    logger = EpochLogger()
    returns, c_returns = [], []
    normalized_returns, normalized_c_returns = [], []


    
    for n in range(num_episodes): 
        true_ep_len = max_ep_len

        if fault_change_step is None: # random changesteps 

            if faults_number is not None:  # you have to ensure faults_number -1 changes
                if min_steps is not None and max_steps is not None: 
                    changesteps = ordered_constrained_fixedlen_list(max_value=true_ep_len, min_diff=min_steps, max_diff=max_steps, max_len=faults_number - 1)
                else:
                    changesteps = sorted(random.sample(range(1, true_ep_len), faults_number-1))
        
            else: # you do not have to ensure a fixed number of change
                if min_steps is not None and max_steps is not None: 
                    changesteps = ordered_constrained_list(true_ep_len, min_steps, max_steps, adaptive_len)
                    if adaptive_len: 
                        true_ep_len = changesteps[-1]
                        env.ref = adjust_list_length(env.ref, true_ep_len)
                else: 
                    raise ValueError("If fault_change_Step is none, you must provide either faults_number (int) or (min_steps and max_steps)")
        else: # fixed change steps
            changesteps = [fault_change_step * (i + 1) for i in range(int(max_ep_len/fault_change_step) -1 )]
    
        change_step_index = 0
        
        if faults_list is None: # random faults 
            fault = np.array([random.uniform(0, 1) for _ in range(env.system.input_dim)])
            faults_string = np.array2string(fault, formatter={'float_kind': '{:.3f}'.format}) + '_'
        else: 
            fault_index = 0
            fault = np.array(faults_list[fault_index])
            faults_string = str(faults_list[fault_index]) + '_'
        
        # reset the environment 
        o = env.reset(initial_fault = fault, obs_logger = ObserverLogger())
        r, c, d, ep_ret, ep_cost, ep_len = 0, 0, False, 0, 0, 0

        for t in range(true_ep_len): 
            a = get_action(o)
            o, r, d, info = env.step(a)

            ep_ret += r
            ep_cost += info['cost']
            ep_len += 1

            if change_step_index < len(changesteps) and t == changesteps[change_step_index]: 
                if faults_list is None: 
                    fault = np.array([random.uniform(0, 1) for _ in range(env.system.input_dim)])
                    env.set_fault(fault)
                    faults_string += str(fault) + '_'
                else: 
                    fault_index += 1
                    env.set_fault(np.array(faults_list[fault_index]))
                    faults_string += str(faults_list[fault_index]) + '_'             
                change_step_index += 1
        
            logger.store(EpRet = ep_ret, EpCost = ep_cost, EpLen = ep_len)

        if num_episodes == 1: 
            print('Episode %d \t EpRet %.3f \t EpCost %.3f \t EpLen %d'%(n, ep_ret, ep_cost, ep_len))

        returns.append(ep_ret)
        c_returns.append(ep_cost)
        normalized_returns.append(ep_ret/ep_len)
        normalized_c_returns.append(ep_cost/ep_len)

        if render: 
            if save_file_name is None: 
                if fault_change_step is None:
                    save_file_name = 'G' + faults_string + 'Episodes' + str(num_episodes) + '_steps' +  str(max_ep_len) + '_change' + str(fault_change_step) + '_' + str(n)
                else: 
                    save_file_name = 'G' + faults_string + 'Episodes' + str(num_episodes) + '_steps' +  str(max_ep_len) + '_' + str(n)
        
            env.render(save = save, save_name = save_file_name + '.' + str(save_format))

            # compute and plot the tracking error
            output_log_array = np.array(env.obs_logger.output_log)
            ref_array = adjust_list_length(env.ref, len(env.obs_logger.output_log))
            track_error_norm = np.linalg.norm(output_log_array - ref_array, axis = 1)

            plt.figure(figsize = (6, 3.25))
            plt.step(env.obs_logger.time_log, track_error_norm, '-', linewidth = 1.5, label = ' Tracking error')
            plt.plot(env.obs_logger.time_log, [env.tracking_threshold]* len(env.obs_logger.time_log), 'r--', label = 'threshold')
            plt.xlabel('time (s)')
            plt.ylabel('tracking error')
            plt.legend(loc = 'lower right')
            plt.xlim([0, env.obs_logger.time_log[-1]])
            plt.title('Tracking Error')
            if save: 
                save_file_name_trackerr = 'Tracking error' if save_file_name is None else 'Tracking_error_' + save_file_name + '.' + str(save_format)
                dpi = 400 if save_file_name_trackerr.endswith(".png") else None
                plt.savefig(save_file_name_trackerr, dpi=dpi, bbox_inches='tight')
        
            
        if num_episodes == 1 and save: 
            save_text_name_RL = 'RL_' + save_file_name + '.txt' if save_file_name is not None else 'RL_' + faults_string + '_' + str(num_episodes) + 'Ep_' + str(max_ep_len) + '_change' + str(fault_change_step) + '_' + str(n) + '.txt'
            with open(save_text_name_RL, 'a') as f: 
                f.write("Episode \t{}\n".format(n))                    
                f.write("EpRet \t{}\n".format(ep_ret))
                f.write("EpCost \t{}\n".format(ep_cost))
                f.write("EpLen \t{}\n".format(ep_len))
                f.write("Faults \t{}\n".format(faults_string))
                f.write("Final_estimate \t{}\n".format(env.dict_s["fault_estimate"]))
                f.write("\n")

        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpCost', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.dump_tabular()
    
        print_statistics('Returns', returns)
        print_statistics('Constraint violations', c_returns)
        print_statistics('Normalized returns', normalized_returns)
        print_statistics('Normalized constraint violations', normalized_c_returns)

        if save: 
            save_text_name_RL = save_file_name + '.txt' if save_file_name is not None else 'Statistics_' + str(num_episodes) + 'Ep.txt'
            save_statistics_to_file(save_text_name_RL, 'Returns', returns)
            save_statistics_to_file(save_text_name_RL, 'Constraint violations', c_returns)
        
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

    parser.add_argument('--save', '-s', action='store_false')
    parser.add_argument('--save_file_name', '-sfnm', type=str, default=None)
    parser.add_argument('--save_format', '-sfmt', type=str, default='pdf')

    parser.add_argument('--adaptive_len', '-al', action='store_true')

    args = parser.parse_args()

"""
Args: 
    
    fpath (str):            path to the folder where the model is saved
    len (int):              length of an episode
    episodes (int):         number of episodes
    itr (int):              number of the iteration of the model to load
    deterministic (bool):   if True, use the deterministic version of the policy
    norender (bool):        if present, do not render the environment 

    faults_list (list):     list of faults. Example of correct form --faults [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
                            If None, then random faults are generated. 

    fault_change_step (int):    number of steps between fault changes. 
                                If None, then the change steps are randomly sampled. 
    
    faults_number (int):    number of faults to change
    min_steps (int):        minimum number of steps between fault changes. 
                            Needed only if fault_change_step is None. 
    max_steps (int):        maximum number of steps between fault changes. 
                            Needed only if fault_change_step is None. 
    
    save (bool):            if present, the plots are saved
    save_file_name (str):   name of the file where to save the plot
    save_format (str):      format of the file to save the plot 

    adaptive_len (bool):    if present, the length of the episode is expanded to ensure the number of randomly generated faults  
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

if args.fault_change_step is None and (args.faults_number is None and (args.min_steps is None and args.max_steps is None)):
    args.faults_number = len(parsed_faults) 

run_policy(env=env, get_action=get_action, 
           max_ep_len = args.len, num_episodes = args.episodes, render = not(args.norender), 
           faults_list = parsed_faults if args.faults_list is not None else None, fault_change_step = args.fault_change_step,
           faults_number = args.faults_number, min_steps = args.min_steps, max_steps = args.max_steps, 
           adaptive_len = args.adaptive_len, 
           save = args.save, save_file_name = args.save_file_name, save_format = args.save_format)



