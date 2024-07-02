import argparse
import os
import json
from collections import deque
from safepo.common.model import ActorVCritic
from safepo.utils.config import multi_agent_velocity_map, multi_agent_goal_tasks
import numpy as np
import joblib
import torch
import gymnasium
import seaborn as sns
import sys
import pandas as pd
sys.path.append('/root/FIERL/')

from environment.env import Environment
from environment.systems.threetank import ThreeTankSystem
from environment.faultobserver.faultobserver import FaultObserver
from environment.utils import InitialConditionSampler, FaultSampler 
from environment.faultobserver.observerlogger import ObserverLogger



def runs_eval(args):
    eval_dir = args.eval_dir
    eval_episodes = args.eval_episodes
    itr = args.itr
    if itr == 'last':
        # find the state{itr}.pkl with the largest itr
        states = os.listdir(eval_dir)
        states = [state for state in states if state.startswith('state') and state.endswith('.pkl')]
        itrs = [int(state.split('state')[1].split('.pkl')[0]) for state in states]
        itr = max(itrs)
    else:
        itr = int(itr)
    


    config_path = eval_dir + '/config.json'
    config = json.load(open(config_path, 'r'))

    # upload environment, add render
    state_env = joblib.load(os.path.join(eval_dir, 'state'+str(itr)+'.pkl'))
    # eval_env = state_env['env'] 
    eval_env = state_env['Env']
    eval_env.is_train = False
    # add observer logger to the environment to plot trajectories
    if args.render: 
        eval_env.observer_logger = ObserverLogger()



    max_ep_len = args.ep_len if args.ep_len is not None else eval_env.max_ep_len

    # upload model 
    model_dir = eval_dir + '/torch_save'
    models = os.listdir(model_dir)
    models = [model for model in models if model.endswith('.pt')]
    final_model_name = sorted(models, key=lambda x: int(x.split('model')[1].split('.pt')[0]))[-1]
    model_path = model_dir + '/' + final_model_name

    model = ActorVCritic(
        obs_dim = gymnasium.spaces.utils.flatdim(eval_env.observation_space),
        act_dim = gymnasium.spaces.utils.flatdim(eval_env.action_space),
        hidden_sizes=config['hidden_sizes'],
    )
    print('model_path:', model_path)
    model.actor.load_state_dict(torch.load(model_path))
    

    # main loop
    reward_sum_list = deque(maxlen=50)
    reward_cost_list = deque(maxlen=50)

    for _ in range(eval_episodes): 
        eval_done = False
        eval_obs, _ = eval_env.reset()
        eval_obs = np.expand_dims(eval_obs, axis=0)
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
        eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
        while not eval_done: 
            with torch.no_grad(): 
                act, _, _, _ = model.step(
                    eval_obs, deterministic=True
                )
            eval_obs, reward, terminated, truncated, info = eval_env.step(act.detach().squeeze().cpu().numpy())
            eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32)
            eval_rew += reward
            eval_cost += info['cost']
            eval_len += 1
            eval_done = terminated or truncated or eval_len >= max_ep_len
        reward_sum_list.append(eval_rew)
        reward_cost_list.append(eval_cost)

        if args.render: 
            eval_env.render(save=True, save_path = eval_dir)

            # # plot the last estimate 
            # state_est, fault_est = eval_env.fault_observer.split()

            # estimate = eval_env.fault_observer.estimate
            # dist = np.random.multivariate_normal(estimate.mean.squeeze(), estimate.cov, size=1000)
            # dist = pd.DataFrame(dist)
            # pairplot = sns.pairplot(dist)
            # pairplot.savefig(eval_dir + '/estimate_last.png')


            # mean_state = state_est.mean.squeeze()


            # dist = np.random.multivariate_normal(state_est.mean.squeeze(), state_est.cov, size=1000)
            # dist = pd.DataFrame(dist)
            # pairplot = sns.pairplot(dist)
            # pairplot.savefig(eval_dir + '/state_est_last.png')


            # dist = np.random.multivariate_normal(fault_est.mean.squeeze(), fault_est.cov, size = 1000)
            # dist = pd.DataFrame(dist)
            # pairplot = sns.pairplot(dist)
            # pairplot.savefig(eval_dir + '/fault_est_last.png')


    rew_sum_mean = sum(reward_sum_list) / len(reward_sum_list)
    rew_sum_std = np.std(reward_sum_list)
    rew_cost_mean = sum(reward_cost_list) / len(reward_cost_list)
    rew_cost_std = np.std(reward_cost_list)

    print(f'Iteration: {itr}, Average Reward Sum: {rew_sum_mean}, ± {rew_sum_std}, Average Reward Cost: {rew_cost_mean}, ± {rew_cost_std}')
    
    if args.save_dir is not None: 
        save_dir = args.save_dir
    else:
        save_dir = eval_dir.replace('runs', 'eval')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    output_file = open(f"{save_dir}/eval_result.txt", 'a')
    exp_name = config['exp_name']
    output_file.write(f"After {eval_episodes} episodes evaluation, the {str(config['exp_name'])} in ThreeTank evaluation reward: {rew_sum_mean}±{rew_sum_std}, cost: {rew_cost_mean}±{rew_cost_std} \n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=str, default='', help="the directory of the evaluation")
    parser.add_argument("--eval-episodes", type=int, default=1, help="the number of episodes to evaluate")
    parser.add_argument("--itr", type=str, default='last', help="the iteration number")
    parser.add_argument("--save-dir", type=str, default=None, help="the directory to save the evaluation results")
    parser.add_argument("--ep-len", type=int, default=20, help="the length of the episode")
    parser.add_argument("--render", type=bool, default=False, help="whether to render the evaluation")
    args = parser.parse_args()

    runs_eval(args)