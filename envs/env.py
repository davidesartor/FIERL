import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import rc
from copy import deepcopy
from gym import spaces
from typing import Tuple

from envs.env_utils import * 

rc('font', family='serif', serif='Times New Roman', size=8)

class Env: 
    def __init__(self, system:object, fault_observer: object,
                 # reference to track 
                 ref:np.ndarray, tracking_threshold:float, 
                 # system state initial conditions
                 ic_type:str, ic_mean:np.ndarray, ic_std:np.ndarray, 
                 # system fault initial conditions
                 faults_mode:str = 'random', faults_list:list = None, 
                 # observer random walk covariance 
                 fault_random_walk:float = 1e-3, 
                 # training or deployment mode
                 is_train:bool = True,
                ): 
        
        if faults_mode not in ['random', 'fixed'] or (faults_mode == 'fixed' and faults_list is None):
            raise ValueError('faults_mode must be either random or fixed and faults_list must be provided if faults_mode is fixed')
        
        if ic_type not in ['normal', 'uniform_component', 'uniform_sphere']:
            raise ValueError('ic_type must be either normal, uniform_component or uniform_sphere')
        
        self.system = deepcopy(system)
        self.ic_type = ic_type
        self.ic_mean = ic_mean
        self.ic_std = ic_std

        self.fault_observer = deepcopy(fault_observer)
        self.faults_mode = faults_mode
        self.faults_list = faults_list
        self.fault_random_walk = fault_random_walk

        self.obs_logger = None

        self.ref = ref
        self.tracking_threshold = tracking_threshold

        self.is_train = is_train
    
    def reset(self, initial_fault:np.ndarray = None, initial_state:np.ndarray = None, obs_logger:object = None) -> np.ndarray: 
        """
        Reset the environment to initial_fault and initial_sate if not None, otherwise sample them randomly from their distributions.
        
        Args: 
            initial_fault (np.ndarray):     array of shape (system.input_dim, ) - initial fault
            initial_state (np.ndarray):     array of shape (system.state_dim, ) - initial state
            obs_logger (object):            logger for observations 
        
        Returns: 
            np.ndarray: array of shape (env.state_dim, ) representing the starting state of the environment
        """

        self.step_counter = 0 

        # sampling a fault if not provided of if is_train
        if initial_fault is None or self.is_train:
            initial_fault = self.__sampleFault()
        
        # sampling an initial state if not provided or if is_train
        if initial_state is None or self.is_train: 
            initial_state = self.__sampleState()
        
        self.system.reset(initial_state)
        self.system.set_fault(initial_fault)
        self.fault_observer.reset()

        if obs_logger is not None: 
            self.obs_logger = deepcopy(obs_logger)
            self.obs_logger.log(self.system, self.fault_observer, None)
        
        self.s, self.dict_s = self.__getState()
        return self.s
    
    def step(self, a:np.ndarray, state_noise:np.ndarray = None, output_noise:np.ndarray = None) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Perform a step in the environment by performing a step in the system and updating the fault_observer. 
        Compute reward, cost. 

        Args: 
            a (np.ndarray): vector of shape (self.system.input_dim,) representing the action to perform.
            state_noise (np.ndarray, optional): process noise, if None is randomly sampled.
            output noise (np.ndarray, optional): measurement noise, if None is randomly sampeld.

        Returns: 
            Tuple[np.ndarray, float, bool, dict]: [new_state, reward, done, info]
        """
        # update system, observer and logger
        self.system.step(a.reshape((-1,1)))
        self.fault_observer.update(A=self.system.A, B = self.system.B, control = a.reshape((-1,1)), 
                                   state_noise_cov = np.eye(self.system.state_dim) * self.system.state_noise_std**2,
                                   C = self.system.C, output = self.system.output,
                                   output_noise_cov = np.eye(self.system.output_dim) * self.system.output_noise_std**2,
                                   fault_random_walk = self.fault_random_walk) 
        if self.obs_logger is not None: 
            self.obs_logger.log(self.system, self.fault_observer, a.reshape((-1,1)))
        
        # update environment state
        self.s, self.dict_s = self.__getState()

        # reward, cost
        reward = self.__computeReward()
        info = {'cost': self.__computeCost()}
        done = False

        self.step_counter += 1

        return self.s, reward, done, info 

        
    def __getState(self, output_noise = None) -> Tuple[np.ndarray, dict]: 
        """
        Define the state of the environment.
        
        Args: 
            output_noise (np.ndarray):  state of the environment as a vector and as a dictionary
        
        Returns: 
            Tuple[np.ndarray, dict]: state of the environment as a vector and as a dictionary
        """
        dict_s = dict()
        dict_s['state_estimate'] = (self.fault_observer.state_estimate.mean, np.triu(self.fault_observer.state_estimate.cov))
        dict_s['fault_estimate'] = (self.fault_observer.fault_estimate.mean, np.triu(self.fault_observer.fault_estimate.cov))
        dict_s['ref'] = self.ref[self.step_counter]
        dict_s['system_output'] = self.system.output
        s = flatten_dictionary(dict_s)
        return s, dict_s

    def __computeReward(self) -> float: 
        return negative_expected_error_true(self.system.fault.reshape(-1,1), self.fault_observer.fault_estimate.mean, self.fault_observer.fault_estimate.cov)
    
    def __computeCost(self) -> int: 
        c = 0
        if np.linalg.norm(self.system.output - self.ref[self.step_counter]) > self.tracking_threshold:
            c = 1
        return c 

    def render(self, save:bool = False, save_name:str = None, title_prefix:str = None):
        if self.obs_logger is not None: 
            self.obs_logger.plot(save, save_name, title_prefix)
        else: 
            print("No logger has been provided.") 
    
    def set_fault(self, new_fault:np.ndarray): 
        self.system.set_fault(new_fault)  

    def __sampleFault(self): 
        if self.faults_mode == 'random': 
            return np.array([[random.uniform(0, 1) for _ in range(1)] for _ in range(self.system.input_dim)]).flatten()
        elif self.faults_mode == 'fixed':
            return np.array(random.choice(self.faults_list)).flatten()
        else:
            raise ValueError('faults_mode must be either random or fixed')
    
    def __sampleState(self): 
            if self.ic_type == 'normal': 
                return np.random.normal(self.ic_mean, self.ic_std)
            elif self.ic_type == 'uniform_component': 
                return np.array([random.uniform(self.ic_mean[i]-self.ic_std,self.ic_mean[i]+self.ic_std ) for i in range(self.system.state_dim)])
            elif self.ic_type == 'uniform_sphere': 
                return sample_uniformly_from_ball(self.ic_mean, self.ic_std, self.system.state_dim)
            else: 
                raise ValueError('ic_type must be either normal, uniform_component or uniform_sphere')
    
    def __getState__(self): 
        return self.__dict__.copy()
    
    def __setState__(self, state): 
        self.__dict__.update(state)

    @property
    def observation_space(self): 
        return spaces.Box(low = -np.inf, high = np.inf, shape = (np.shape(self.s)[0],), dtype=np.float64)
    
    @property 
    def action_space(self): 
        return spaces.Box(low = self.system.min_input, high= self.system.max_input, shape = (self.system.input_dim,), dtype = np.float64)
