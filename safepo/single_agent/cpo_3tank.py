from __future__ import annotations
import gymnasium as gym

import os
import random
import sys
import time
from collections import deque
from typing import Callable
from distutils.util import strtobool


import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import single_agent_args, isaac_gym_map, parse_sim_params



from environment.env import Environment
from environment.systems.threetank import ThreeTankSystem
from environment.faultobserver.faultobserver import FaultObserver
from environment.utils import InitialConditionSampler, FaultSampler 

import argparse

import wandb
wandb.login()


STEP_FRACTION=0.8
CPO_SEARCHING_STEPS=15
CONJUGATE_GRADIENT_ITERS=15


default_cfg = {
    'hidden_sizes': [512, 512, 512],
    'gamma': 0.99,
    'target_kl': 0.01,
    'batch_size': 128,
    'learning_iters': 10,
    'max_grad_norm': 40.0,
    'steps_per_epoch': 4000, 
    'total_steps': 4000*1000,
    'task': None,
    'cost_limit': 3,
    'log_std_param': False, # layer or parameter the learned log std 
}

def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, "No gradients were found in model parameters."
    return torch.cat(flat_params)


def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x, policy, fvp_obs)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p, policy, fvp_obs)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x


def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f"Lengths do not match: {i} vs. {len(vals)}"

def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, "No gradients were found in model parameters."
    return torch.cat(grads)

def fvp(
    params: torch.Tensor,
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
) -> torch.Tensor:
    policy.actor.zero_grad()
    current_distribution = policy.actor(fvp_obs)
    with torch.no_grad():
        old_distribution = policy.actor(fvp_obs)
    kl = torch.distributions.kl.kl_divergence(
        old_distribution, current_distribution
    ).mean()

    grads = torch.autograd.grad(kl, tuple(policy.actor.parameters()), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_p = (flat_grad_kl * params).sum()
    grads = torch.autograd.grad(
        kl_p,
        tuple(policy.actor.parameters()),
        retain_graph=False,
    )

    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    return flat_grad_grad_kl + params * 0.1


def main(args, cfg_env=None):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(f'{args.device}:{args.device_id}')
    
    config = default_cfg

    # with wandb.init(project='FIERL_3tank_obs_new', config = config):
    if True:

        # environment definition
        max_ep_len = 20 
        tracking_threshold = 0.1
        reference = [np.array([[0.489], [0.2332]])] * max_ep_len

        min_action = -0.002
        max_action = 0.02
        system = ThreeTankSystem(state_noise_std=1e-5, output_noise_std=1e-5, min_input=min_action, max_input = max_action)
        fault_observer = FaultObserver(state_dim = system.state_dim, input_dim = system.input_dim,  fault_evol_cov = 1e-5)
        # FaultObserver(initial_state_estimate=system.state_dim, initial_fault_estimate=system.input_dim)

        ic_sampling_fnc = InitialConditionSampler(ic_type='ncube', half_side=1e-3, center = np.array([0.489, 0.2332, 0.3611]), n_samples=1)
        fault_sampling_fnc = FaultSampler(fault_type='uniform', a=0, b=1, size=(system .input_dim,))

        env = Environment(system = system, 
                        fault_observer = fault_observer, 
                        reference = reference, 
                        track_threshold = tracking_threshold, 
                        initial_condition_fnc = ic_sampling_fnc,
                        fault_generator_fnc = fault_sampling_fnc,
                        max_ep_len = max_ep_len,)
        act_space = env.action_space
        obs_space = env.observation_space

        # set training steps
        steps_per_epoch = config.get("steps_per_epoch", args.steps_per_epoch)
        total_steps = config.get("total_steps", args.total_steps)
        local_steps_per_epoch = steps_per_epoch // args.num_envs
        epochs = total_steps // steps_per_epoch

        # create the actor-critic module
        policy = ActorVCritic(
            obs_dim=gym.spaces.utils.flatdim(obs_space),
            act_dim=act_space.shape[0],
            hidden_sizes=config["hidden_sizes"],
            log_std_params=config["log_std_param"],
        ).to(device)
        reward_critic_optimizer = torch.optim.Adam(
            policy.reward_critic.parameters(), lr=1e-3
        )
        cost_critic_optimizer = torch.optim.Adam(
            policy.cost_critic.parameters(), lr=1e-3
        )

        # create the vectorized on-policy buffer
        buffer = VectorizedOnPolicyBuffer(
            obs_space= gym.spaces.utils.flatten_space(obs_space),
            act_space=act_space,
            size=local_steps_per_epoch,
            device=device,
            num_envs=args.num_envs,
            gamma=config["gamma"],
        )

        # set up the logger
        dict_args = vars(args)
        dict_args.update(config)
        logger = EpochLogger(
            log_dir=args.log_dir,
            seed=str(args.seed),
        )
        rew_deque = deque(maxlen=50)
        cost_deque = deque(maxlen=50)
        len_deque = deque(maxlen=50)
        eval_rew_deque = deque(maxlen=50)
        eval_cost_deque = deque(maxlen=50)
        eval_len_deque = deque(maxlen=50)
        logger.save_config(dict_args)
        logger.setup_torch_saver(policy.actor)
        logger.log("Start with training.")
        obs, _ = env.reset()
        obs = np.expand_dims(obs, axis=0) # modifica
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        ep_ret, ep_cost, ep_len = (
            np.zeros(args.num_envs),
            np.zeros(args.num_envs),
            np.zeros(args.num_envs),
        )
        # training loop
        # wandb.watch(policy, log='all', log_freq = 10)
        for epoch in range(epochs):
            rollout_start_time = time.time()
            # collect samples until we have enough to update
            for steps in range(local_steps_per_epoch):
                with torch.no_grad():
                    act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
                action = act.detach().squeeze() if args.task in isaac_gym_map.keys() else act.detach().squeeze().cpu().numpy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                cost = info["cost"] 
                reward = np.expand_dims(reward, axis=0) # modifica
                cost = np.expand_dims(cost, axis=0) # modifica
                terminated = np.expand_dims(terminated, axis=0) # modifica
                truncated = np.expand_dims(truncated, axis=0) # modifica

                print('oook here==============================')

        
                ep_ret += reward.cpu().numpy() if args.task in isaac_gym_map.keys() else reward
                ep_cost += cost.cpu().numpy() if args.task in isaac_gym_map.keys() else cost
                ep_len += 1
                next_obs, reward, cost, terminated, truncated = (
                    torch.as_tensor(x, dtype=torch.float32, device=device)
                    for x in (next_obs, reward, cost, terminated, truncated)
                )
                if "final_observation" in info:
                    info["final_observation"] = np.array(
                        [
                            array if array is not None else np.zeros(obs.shape[-1])
                            for array in info["final_observation"]
                        ],
                    )
                    info["final_observation"] = torch.as_tensor(
                        info["final_observation"],
                        dtype=torch.float32,
                        device=device,
                    )

                buffer.store(
                    obs=obs,
                    act=act,
                    reward=reward,
                    cost=cost,
                    value_r=value_r,
                    value_c=value_c,
                    log_prob=log_prob,
                )

                obs = next_obs
                obs = torch.Tensor(np.expand_dims(obs, axis=0)) #modifica
                epoch_end = steps >= local_steps_per_epoch - 1
                for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                    if epoch_end or done or time_out:
                        last_value_r = torch.zeros(1, device=device)
                        last_value_c = torch.zeros(1, device=device)
                        if not done:
                            if epoch_end:
                                with torch.no_grad():
                                    _, _, last_value_r, last_value_c = policy.step(
                                        obs[idx], deterministic=False
                                    )
                            if time_out:
                                with torch.no_grad():
                                    final_observation = torch.Tensor(np.expand_dims(info["final_observation"], axis=0))[idx] # modifica
                                    _, _, last_value_r, last_value_c = policy.step(
                                        final_observation, deterministic=False #modifica
                                    )
                            last_value_r = last_value_r.unsqueeze(0)
                            last_value_c = last_value_c.unsqueeze(0)
                        if done or time_out:
                            rew_deque.append(ep_ret[idx])
                            cost_deque.append(ep_cost[idx])
                            len_deque.append(ep_len[idx])
                            logger.store(
                                **{
                                    "Metrics/EpRet": np.mean(rew_deque),
                                    "Metrics/EpCost": np.mean(cost_deque),
                                    "Metrics/EpLen": np.mean(len_deque),
                                }
                            )
                            # wandb.log({'EpRet': np.mean(rew_deque), 'EpCost': np.mean(cost_deque)},  step = epoch)
                            ep_ret[idx] = 0.0
                            ep_cost[idx] = 0.0
                            ep_len[idx] = 0.0
                            logger.logged = False
                        
                        buffer.finish_path(
                            last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                        )
                        # reset environment
                        obs, _ = env.reset() # modifica
                        obs = torch.Tensor(np.expand_dims(obs, axis=0)) # modifica
                    
            rollout_end_time = time.time()

            eval_start_time = time.time()

            eval_episodes = 1 if epoch < epochs - 1 else 10
            if args.use_eval:
                for _ in range(eval_episodes):
                    eval_done = False
                    eval_obs, _ = env.reset()
                    eval_obs = np.expand_dims(eval_obs, axis=0) # modifica
                    eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                    eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                    while not eval_done:
                        with torch.no_grad():
                            act, log_prob, value_r, value_c = policy.step(eval_obs, deterministic=True)
                        next_obs, reward, terminated, truncated, info = env.step(
                            act.detach().squeeze().cpu().numpy()
                        )
                        cost = info["cost"]
                        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                        eval_rew += reward
                        eval_cost += cost
                        eval_len += 1
                        eval_done = terminated[0] or truncated[0]
                        eval_obs = next_obs
                    eval_rew_deque.append(eval_rew)
                    eval_cost_deque.append(eval_cost)
                    eval_len_deque.append(eval_len)
                logger.store(
                    **{
                        "Metrics/EvalEpRet": np.mean(eval_rew),
                        "Metrics/EvalEpCost": np.mean(eval_cost),
                        "Metrics/EvalEpLen": np.mean(eval_len),
                    }
                )

            eval_end_time = time.time()

            # update policy
            data = buffer.get()
            fvp_obs = data["obs"][:: 1]
            theta_old = get_flat_params_from(policy.actor)
            policy.actor.zero_grad()
            # compute loss_pi
            temp_distribution = policy.actor(data["obs"])
            log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
            ratio = torch.exp(log_prob - data["log_prob"])
            loss_pi_r = -(ratio * data["adv_r"]).mean()
            loss_reward_before = loss_pi_r.item()
            old_distribution = policy.actor(data["obs"])

            loss_pi_r.backward()

            grads = -get_flat_gradients_from(policy.actor)
            x = conjugate_gradients(fvp, policy, fvp_obs, grads, CONJUGATE_GRADIENT_ITERS)
            assert torch.isfinite(x).all(), "x is not finite"
            xHx = torch.dot(x, fvp(x, policy, fvp_obs))
            assert xHx.item() >= 0, "xHx is negative"
            alpha = torch.sqrt(2 * config['target_kl'] / (xHx + 1e-8))

            policy.actor.zero_grad()
            temp_distribution = policy.actor(data["obs"])
            log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
            ratio = torch.exp(log_prob - data["log_prob"])
            loss_pi_c = (ratio * data["adv_c"]).mean()
            loss_cost_before = loss_pi_c.item()

            loss_pi_c.backward()

            b_grads = get_flat_gradients_from(policy.actor)
            ep_costs = logger.get_stats("Metrics/EpCost") - args.cost_limit

            p = conjugate_gradients(fvp, policy, fvp_obs, b_grads, CONJUGATE_GRADIENT_ITERS)
            q = xHx
            r = grads.dot(p)
            s = b_grads.dot(p)

            if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
                A = torch.zeros(1)
                B = torch.zeros(1)
                optim_case = 4
            else:
                assert torch.isfinite(r).all(), "r is not finite"
                assert torch.isfinite(s).all(), "s is not finite"

                A = q - r**2 / (s + 1e-8)
                B = 2 * config['target_kl'] - ep_costs**2 / (s + 1e-8)

                if ep_costs < 0 and B < 0:
                    optim_case = 3
                elif ep_costs < 0 <= B:
                    optim_case = 2
                elif ep_costs >= 0 and B >= 0:
                    optim_case = 1
                    logger.log("Alert! Attempting feasible recovery!", "yellow")
                else:
                    optim_case = 0
                    logger.log("Alert! Attempting infeasible recovery!", "red")

            if optim_case in (3, 4):
                alpha = torch.sqrt(2 * config['target_kl'] / (xHx + 1e-8))
                nu_star = torch.zeros(1)
                lambda_star = 1 / (alpha + 1e-8)
                step_direction = alpha * x

            elif optim_case in (1, 2):

                def project(
                    data: torch.Tensor, low: torch.Tensor, high: torch.Tensor
                ) -> torch.Tensor:
                    """Project data to [low, high] interval."""
                    return torch.clamp(data, low, high)

                lambda_a = torch.sqrt(A / B)
                lambda_b = torch.sqrt(q / (2 * config['target_kl']))
                r_num = r.item()
                eps_cost = ep_costs + 1e-8
                if ep_costs < 0:
                    lambda_a_star = project(
                        lambda_a, torch.as_tensor(0.0), r_num / eps_cost
                    )
                    lambda_b_star = project(
                        lambda_b, r_num / eps_cost, torch.as_tensor(torch.inf)
                    )
                else:
                    lambda_a_star = project(
                        lambda_a, r_num / eps_cost, torch.as_tensor(torch.inf)
                    )
                    lambda_b_star = project(
                        lambda_b, torch.as_tensor(0.0), r_num / eps_cost
                    )

                def f_a(lam: torch.Tensor) -> torch.Tensor:
                    return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

                def f_b(lam: torch.Tensor) -> torch.Tensor:
                    return -0.5 * (q / (lam + 1e-8) + 2 * config['target_kl'] * lam)

                lambda_star = (
                    lambda_a_star
                    if f_a(lambda_a_star) >= f_b(lambda_b_star)
                    else lambda_b_star
                )

                nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)

                step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

            else:
                lambda_star = torch.zeros(1)
                nu_star = torch.sqrt(2 * config['target_kl'] / (s + 1e-8))
                step_direction = -nu_star * p

            step_frac = 1.0
            theta_old = get_flat_params_from(policy.actor)
            expected_reward_improve = grads.dot(step_direction)

            kl = torch.zeros(1)
            for step in range(CPO_SEARCHING_STEPS):
                new_theta = theta_old + step_frac * step_direction
                set_param_values_to_model(policy.actor, new_theta)
                acceptance_step = step + 1

                with torch.no_grad():
                    try:
                        temp_distribution = policy.actor(data["obs"])
                        log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                        ratio = torch.exp(log_prob - data["log_prob"])
                        loss_reward = -(ratio * data["adv_r"]).mean()
                    except ValueError:
                        step_frac *= STEP_FRACTION
                        continue
                    temp_distribution = policy.actor(data["obs"])
                    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                    ratio = torch.exp(log_prob - data["log_prob"])
                    loss_cost = (ratio * data["adv_c"]).mean()
                    current_distribution = policy.actor(data["obs"])
                    kl = torch.distributions.kl.kl_divergence(
                        old_distribution, current_distribution
                    ).mean()
                loss_reward_improve = loss_reward_before - loss_reward.item()
                loss_cost_diff = loss_cost.item() - loss_cost_before

                logger.log(
                    f"Expected Improvement: {expected_reward_improve} Actual: {loss_reward_improve}",
                )
                if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                    logger.log("WARNING: loss_pi not finite")
                if not torch.isfinite(kl):
                    logger.log("WARNING: KL not finite")
                    continue
                if loss_reward_improve < 0 if optim_case > 1 else False:
                    logger.log("INFO: did not improve improve <0")
                elif loss_cost_diff > max(-ep_costs, 0):
                    logger.log(f"INFO: no improve {loss_cost_diff} > {max(-ep_costs, 0)}")
                elif kl > config["target_kl"]:
                    logger.log(f"INFO: violated KL constraint {kl} at step {step + 1}.")
                else:
                    logger.log(f"Accept step at i={step + 1}")
                    break
                step_frac *= STEP_FRACTION
            else:
                logger.log("INFO: no suitable step found...")
                step_direction = torch.zeros_like(step_direction)
                acceptance_step = 0

            theta_new = theta_old + step_frac * step_direction
            set_param_values_to_model(policy.actor, theta_new)

            logger.store(
                **{
                    "Misc/Alpha": alpha.item(),
                    "Misc/FinalStepNorm": torch.norm(step_direction).mean().item(),
                    "Misc/xHx": xHx.item(),
                    "Misc/gradient_norm": torch.norm(grads).mean().item(),
                    "Misc/H_inv_g": x.norm().item(),
                    "Misc/AcceptanceStep": acceptance_step,
                    "Loss/Loss_actor": (loss_pi_r + loss_pi_c).mean().item(),
                    "Train/KL": kl.cpu(),
                },
            )

            dataloader = DataLoader(
                dataset=TensorDataset(
                    data["obs"],
                    data["target_value_r"],
                    data["target_value_c"],
                ),
                batch_size=config.get("batch_size", args.steps_per_epoch//config.get("num_mini_batch", 1)),
                shuffle=True,
            )
            for _ in range(config["learning_iters"]):
                for (
                    obs_b,
                    target_value_r_b,
                    target_value_c_b,
                ) in dataloader:
                    reward_critic_optimizer.zero_grad()
                    loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
                    cost_critic_optimizer.zero_grad()
                    loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                    if config.get("use_critic_norm", True):
                        for param in policy.reward_critic.parameters():
                            loss_r += param.pow(2).sum() * 0.001
                        for param in policy.cost_critic.parameters():
                            loss_c += param.pow(2).sum() * 0.001
                    total_loss = 2*loss_r + loss_c \
                        if config.get("use_value_coefficient", False) \
                        else loss_r + loss_c
                    total_loss.backward()
                    clip_grad_norm_(policy.parameters(), config["max_grad_norm"])
                    reward_critic_optimizer.step()
                    cost_critic_optimizer.step()

                    logger.store(
                        **{
                            "Loss/Loss_reward_critic": loss_r.mean().item(),
                            "Loss/Loss_cost_critic": loss_c.mean().item(),
                        }
                    )
            update_end_time = time.time()
            if not logger.logged:
                # log data
                logger.log_tabular("Metrics/EpRet")
                logger.log_tabular("Metrics/EpCost")
                logger.log_tabular("Metrics/EpLen")
                if args.use_eval:
                    logger.log_tabular("Metrics/EvalEpRet")
                    logger.log_tabular("Metrics/EvalEpCost")
                    logger.log_tabular("Metrics/EvalEpLen")
                logger.log_tabular("Train/Epoch", epoch + 1)
                logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
                logger.log_tabular("Train/KL")
                logger.log_tabular("Loss/Loss_reward_critic")
                logger.log_tabular("Loss/Loss_cost_critic")
                logger.log_tabular("Loss/Loss_actor")
                logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
                if args.use_eval:
                    logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
                logger.log_tabular("Time/Update", update_end_time - eval_end_time)
                logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
                logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
                logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())
                logger.log_tabular("Misc/Alpha")
                logger.log_tabular("Misc/FinalStepNorm")
                logger.log_tabular("Misc/xHx")
                logger.log_tabular("Misc/gradient_norm")
                logger.log_tabular("Misc/H_inv_g")
                logger.log_tabular("Misc/AcceptanceStep")

                logger.dump_tabular()
            


                if (epoch+1) % 100 == 0 or epoch == 0 or epoch == epochs - 1:
                    logger.torch_save(itr=epoch)
                    if args.task not in isaac_gym_map.keys():
                        logger.save_state(
                            state_dict={
                                # "Normalizer": env.obs_rms,# modifica
                                'env': env, # modifica
                            },
                            itr = epoch
                        )
        logger.close()

def single_agent_args_3tank():
    custom_parameters = [
        {"name": "--seed", "type": int, "default":0, "help": "Random seed"},
        {"name": "--use-eval", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Use evaluation environment for testing"},
        {"name": "--task", "type": str, "default": "", "help": "The task to run"},
        {"name": "--num-envs", "type": int, "default": 1, "help": "The number of parallel game environments"},
        {"name": "--experiment", "type": str, "default": "3T_observer_new", "help": "Experiment name"},
        {"name": "--log-dir", "type": str, "default": "runs", "help": "directory to save agent logs"},
        {"name": "--device", "type": str, "default": "cpu", "help": "The device to run the model on"},
        {"name": "--device-id", "type": int, "default": 0, "help": "The device id to run the model on"},
        {"name": "--write-terminal", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Toggles terminal logging"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Toggles headless mode"},
        {"name": "--total-steps", "type": int, "default": 10000000, "help": "Total timesteps of the experiments"},
        {"name": "--steps-per-epoch", "type": int, "default": 20000, "help": "The number of steps to run in each environment per policy rollout"},
        {"name": "--randomize", "type": bool, "default": False, "help": "Wheather to randomize the environments' initial states"},
        {"name": "--cost-limit", "type": float, "default": 3.0, "help": "cost_lim"},
        {"name": "--lagrangian-multiplier-init", "type": float, "default": 0.001, "help": "initial value of lagrangian multiplier"},
        {"name": "--lagrangian-multiplier-lr", "type": float, "default": 0.035, "help": "learning rate of lagrangian multiplier"},
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    cfg_env={}
    base_path = os.path.dirname(os.path.abspath(__file__)).replace("utils", "multi_agent")
    return args, cfg_env

if __name__ == "__main__":
    args, cfg_env = single_agent_args_3tank()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
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
                main(args, cfg_env)
    else:
        main(args, cfg_env)


