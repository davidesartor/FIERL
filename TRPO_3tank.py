import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import gymnasium as gym 
import scipy.signal
from collections import namedtuple

from environment.env import Environment
from environment.systems.threetank import ThreeTankSystem
from environment.faultobserver.faultobserver import FaultObserver
from environment.utils import InitialConditionSampler, FaultSampler 

from torchviz import make_dot


# Environment 
max_ep_len = 20 
tracking_threshold = 0.1
reference = [np.array([[0.489], [0.2332]])] * max_ep_len

min_action = None #-0.02
max_action = None #0.02
system = ThreeTankSystem(state_noise_std=0.001, output_noise_std=0.001, min_input=min_action, max_input = max_action)
fault_observer = FaultObserver(initial_state_estimate=system.state_dim, initial_fault_estimate=system.input_dim)

ic_sampling_fnc = InitialConditionSampler(ic_type='ncube', half_side=0.005, center = np.array([0.489, 0.2332, 0.3611]), n_samples=1)
fault_sampling_fnc = FaultSampler(fault_type='uniform', a=0, b=1, size=(system .input_dim,))

env = Environment(system = system, 
                  fault_observer = fault_observer, 
                  reference = reference, 
                  track_threshold = tracking_threshold, 
                  initial_condition_fnc = ic_sampling_fnc,
                  fault_generator_fnc = fault_sampling_fnc,
                  fault_random_walk_std = 0.001,
                  is_train = True,)

state_dim = env.env_obs_shape[0]
print('state_dim', state_dim)
action_dim = env.action_space.shape[0]


# ROLLOUT
Rollout = namedtuple('Rollout', ['states', 'actions', 'rewards', 'next_states', 'truncateds'])


# TRAIN AGENT 

def train(epochs = 100, num_rollouts = 10, render_frequency = None): 
    mean_total_rewards = []
    global_rollout = 0 

    for epoch in range(epochs): 
        rollouts = []
        rollout_total_rewards = []

        for t in range(num_rollouts): 
            state = env.reset()[0] # shape (state_size, )
            reward, terminated, truncated, ep_len = 0, False, False, 0
            
            samples = []

            while not terminated and not truncated and ep_len < max_ep_len:
                if render_frequency is not None and global_rollout % render_frequency == 0: 
                    env.render()
                
                with torch.no_grad(): 
                    action = get_action(state)
                    # print('this is action', action, '---------------------------------------------')
                
                next_state, reward, terminated, truncated, info = env.step(action)
                reward = reward.item() # convert from numpy to scalar
                truncated = True if ep_len == max_ep_len - 1 else False

                # Collect samples 
                samples.append((state, action, reward, next_state, truncated))

                state = next_state
                ep_len += 1
            
            # Transpose samples 
            states, actions, rewards, next_states, truncateds = zip(*samples)

            states = torch.stack([torch.from_numpy(state) for state in states], dim=0).float()
            next_states = torch.stack([torch.from_numpy(state) for state in next_states], dim=0).float()
            actions = torch.stack([torch.from_numpy(action) for action in actions], dim=0).float()
            rewards = torch.as_tensor(rewards).unsqueeze(1)
            truncateds = torch.as_tensor(truncateds).unsqueeze(1)
            
            rollouts.append(Rollout(states, actions, rewards, next_states, truncateds))

            rollout_total_rewards.append(rewards.sum().item())
            global_rollout += 1

        update_agent(rollouts)
        mtr = np.mean(rollout_total_rewards)
        print(f'E: {epoch}.\tMean total reward across {num_rollouts} rollouts: {mtr}')

        mean_total_rewards.append(mtr)

    plt.plot(mean_total_rewards)
    plt.show()



# ACTOR
class GaussianMLP(nn.Module): 
    '''
    Gaussian policy network

    Args: 
        input_dim (int): input dimension
        output_dim (int): output dimension
        hidden_sizes (list): hidden layer sizes
        activation (nn.Module): activation function
    '''
    def __init__(self, state_dim, action_dim, hidden_sizes, activation = nn.Tanh): 
        super().__init__()

        layers = []
        self.input_dim = state_dim
        self.output_dim = action_dim

        for i in range(len(hidden_sizes)): 
            layers.append(nn.Linear(state_dim, hidden_sizes[i]))
            layers.append(activation())
            state_dim = hidden_sizes[i]

        layers.append(nn.Linear(state_dim, action_dim))
        self.mean = nn.Sequential(*layers)
        self.log_std = nn.Parameter(0.5 * torch.ones(action_dim))
    
    def forward(self, state):
        return self.mean(state), self.log_std


def get_action(state): 
    state = torch.tensor(state).float().unsqueeze(0) # Turn into a batch with single element, as required by nn.Module
    mean, log_std = actor(state)
    sampled_action = mean + torch.exp(log_std) * torch.randn_like(mean)
    return sampled_action.numpy()[0]

actor = GaussianMLP(state_dim = state_dim, action_dim = action_dim, hidden_sizes = [32, 32])


# CRITICS 
class ValueFunction(nn.Module): 
    def __init__(self, state_dim, hidden_sizes, activation = nn.ReLU): 
        super().__init__()

        layers = []
        self.input_dim = state_dim

        for i in range(len(hidden_sizes)): 
            layers.append(nn.Linear(state_dim, hidden_sizes[i]))
            layers.append(activation())
            state_dim = hidden_sizes[i]

        layers.append(nn.Linear(state_dim, 1))
        self.value = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.value(state)
    

critic = ValueFunction(state_dim = state_dim, hidden_sizes = [32, 32])

# critic = nn.Sequential(nn.Linear(state_dim, 32), 
#                        nn.ReLU(), 
#                     #    nn.Linear(32, 32), 
#                     #    nn.ReLU(), 
#                        nn.Linear(32, 1))
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=0.005) 

# cvf_critic = nn.Sequential(nn.Linear(state_dim, 32),
#                         nn.ReLU(), 
#                         nn.Linear(32, 1))
# cvf_critic_optimizer = torch.optim.Adam(cvf_critic.parameters(), lr=0.005)

def update_critic(advantages): 
    loss = 0.5 * (advantages ** 2).mean() # MSE
    critic_optimizer.zero_grad()
    loss.backward()
    critic_optimizer.step()

def update_critic_returns(states, returns): 
    returns1 = returns.detach()
    critic_optimizer.zero_grad()  
    # a = critic(states) - returns1
    # a = (a-a.mean())/a.std()
    # loss = 0.5 * a.pow(2).mean()
    loss = 0.5 * (critic(states) - returns1).pow(2).mean()
    loss.backward() #backward(retain_graph=True)
    critic_optimizer.step()


# UPDATE AGENT 
max_d_kl = 0.01


def update_agent(rollouts): 
    states = torch.cat([r.states for r in rollouts], dim=0)
    actions = torch.cat([r.actions for r in rollouts], dim=0)

    advantages = [estimate_advantages(states, next_states[-1], rewards) for states, _, rewards, next_states, _ in rollouts]
    advantages = torch.cat(advantages, dim=0).flatten()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = [estimate_returns(states, next_states[-1], rewards) for states, _, rewards, next_states, _ in rollouts]
    returns = torch.cat(returns, dim=0).flatten()

    # update_critic(advantages)
    for _ in range(80): 
        torch.autograd.set_detect_anomaly(True)
        update_critic_returns(states, returns)

    means, log_stds = actor.forward(states)
    probabilities = torch.exp(gaussian_log_likelihood(actions, means, log_stds))
    L = surrogate_loss(probabilities, probabilities.detach(), advantages)
    KL = gaussian_kl_div(means, torch.exp(log_stds), means, torch.exp(log_stds))

    parameters = list(actor.parameters())

    g = flat_grad(L, parameters, retain_graph=True)
    d_kl = flat_grad(KL, parameters, create_graph=True) 

    def HVP(v): 
        return flat_grad(d_kl @ v, parameters, retain_graph=True)
    
    search_dir = conjugate_gradient(HVP, g)
    max_length = torch.sqrt(2 * max_d_kl / (search_dir @ HVP(search_dir)))
    max_step = max_length * search_dir

    def criterion(step):
        apply_update(step)

        with torch.no_grad():
            new_means, new_log_stds = actor.forward(states)
            new_stds = torch.exp(new_log_stds)
            probabilities_new = torch.exp(gaussian_log_likelihood(actions, new_means, new_log_stds))
            L_new = surrogate_loss(probabilities_new, probabilities, advantages)
            KL_new = gaussian_kl_div(means, torch.exp(log_stds), new_means, new_stds)

        L_improvement = L_new - L

        if L_improvement > 0 and KL_new <= max_d_kl:
            return True

        apply_update(-step)
        return False

    i = 0
    while not criterion((0.9 ** i) * max_step) and i < 10:
        i += 1



def estimate_advantages(states, last_state, rewards):
    values = critic(states)
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
    advantages = next_values - values
    return advantages

def estimate_returns(states, last_state, rewards): 
    last_value = critic(last_state.unsqueeze(0))
    next_values = torch.zeros_like(rewards)
    for i in reversed(range(rewards.shape[0])):
        last_value = next_values[i] = rewards[i] + 0.99 * last_value
    return next_values 

def gaussian_log_likelihood(x, mu, log_std): 
    pre_sum = - 0.5 * (torch.log(2*torch.tensor(np.pi)) + 2 * log_std + (x - mu)**2 / torch.exp(log_std)**2)
    return pre_sum.sum(dim=1)


def discount_cumsum(x, discount): 
    ''' From rllab, to compute discounted cumulative sums of vectors. 
    Args: 
        x (np.ndarray): vector to be summed e.g., 
        [x0, 
         x1, 
         x2]
        discount (float): discount factor
    
    Returns: 
        discounted_cumsum (np.ndarray): 
        [x0 + discount * x1 + discount^2 * x2, 
         x1 + discount * x2, 
         x2]
    '''
    x = x.detach().numpy()
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def surrogate_loss(new_probabilities, old_probabilities, advantages):
    return (new_probabilities / old_probabilities * advantages).mean()


def gaussian_kl_div(mean1, std1, mean2, std2):
    # mean1 = mean1.detach()
    # std1 = std1.detach()
    mean2 = mean2.detach()
    std2 = std2.detach()

    mean_diff = mean1 - mean2
    var1 = std1 ** 2
    var2 = std2 ** 2

    return (0.5 * (torch.log(var2 / var1) + (var1 + mean_diff ** 2) / var2 - 1)).mean()

def kl_div(p, q):
    p = p.detach()
    return (p * (p.log() - q.log())).sum(-1).mean()


def flat_grad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True

    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


def conjugate_gradient(A, b, delta=0., max_iterations=10):
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()

    i = 0
    while i < max_iterations:
        AVP = A(p)

        dot_old = r @ r
        alpha = dot_old / (p @ AVP)

        x_new = x + alpha * p

        if (x - x_new).norm() <= delta:
            return x_new

        i += 1
        r = r - alpha * AVP

        beta = (r @ r) / dot_old
        p = r + beta * p

        x = x_new
    return x


def apply_update(grad_flattened):
    n = 0
    for p in actor.parameters():
        numel = p.numel()
        g = grad_flattened[n:n + numel].view(p.shape)
        p.data += g
        n += numel

# train agent 
train(epochs=100, num_rollouts=200, render_frequency=None)
