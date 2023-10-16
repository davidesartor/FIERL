import os 
import sys
import warnings
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
warnings.filterwarnings("ignore",category=FutureWarning)

import safe_rl
from safe_rl.utils.run_utils import setup_logger_kwargs
from safe_rl.utils.mpi_tools import mpi_fork
from safe_rl.pg.algos import cpo

from envs.systems.threetanksys import ThreeTankSystem
from envs.faultobserver.faultobserver import FaultObserver
from envs.env import Env


save_freq = 10
exp_name = "3Tank"
seed = np.random.randint(1,100)

# number of policy update to run
epochs = 1000

# length of each training episode
max_ep_len = 40    

# number of steps per policy update (number of episodes * max_ep_len)
steps_per_epoch = 3600

# cost limit d on the constraint return 
cost_lim = max_ep_len * 0.15

# system, environment and fault observer 
faults_list = None

# initial condition, abuse of notation: 
    # if normal x0 = N(x0_mean, x0_std**2)
    # if uniform_sphere x0_mean is the center of the ball, x0_std is the radius of the ball
    # if uniform x0 = U(x0_mean - x0_std, x0_mean + x0_std)
x0_mean = np.array([[0.489], [0.2332], [0.3611]])
x0_std = 0.1
ic_type = 'uniform_sphere'

# action limits
min_action = -0.002
max_action = 0.02

# reference trajectory and admissible tracking error
reference = [np.array([[0.489], [0.2332]])] * max_ep_len
track_t = 0.1

sys = ThreeTankSystem(state_noise_std=1e-3, output_noise_std=1e-3, min_input=min_action, max_input=max_action)
faultObserver = FaultObserver(initial_state_estimate= sys.state_dim, initial_fault_estimate=sys.input_dim)

env = Env(system=sys, 
          fault_observer=faultObserver,
          ref=reference,
          tracking_threshold=track_t,
          ic_type=ic_type,
          ic_mean = x0_mean,
          ic_std = x0_std, 
          faults_mode = 'random',
          faults_list = faults_list, 
          fault_random_walk=1e-3,
          is_train=True,
          )

cpo(
    env_fn = env,
    ac_kwargs = dict(hidden_sizes=(64, 64, 64)),
    logger_kwargs = setup_logger_kwargs(exp_name, seed),
    cost_lim = cost_lim,
    save_freq = save_freq,
    max_ep_len = max_ep_len,
    steps_per_epoch = steps_per_epoch,
    vf_iters = 80,
    vf_lr = 1e-3,
    seed = seed,
    epochs = epochs,
    squashed_policy = True,
    policy_lb = min_action,
    policy_ub = max_action,
)


