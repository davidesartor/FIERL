import numpy as np
from copy import deepcopy
import gymnasium as gym 
from gymnasium import spaces 
import inspect
from environment.utils import * 


class Environment(gym.Env): 
    
    def __init__(self, system, fault_observer, reference, track_threshold, initial_condition_fnc, fault_generator_fnc, fault_random_walk_std = None, is_train = True,  render_mode = None): 
        '''
        Args: 
            system (object): system
            fault_observer (object): fault observer
            reference (np.ndarray): reference signal
            track_threshold (float): threshold for tracking error
            initial_condition_fnc (callable): function to generate initial conditions of the system 
            fault_generator_fnc (callable): function to generate faults
            fault_random_walk_std (float): standard deviation of the random walk of the fault
            is_train (bool): whether the environment is in training mode
            render_mode (str): render mode
        '''
        self.system = deepcopy(system)
        self.fault_observer = deepcopy(fault_observer)

        self.reference = reference
        self.track_threshold = track_threshold

        self.initial_condition_fnc = initial_condition_fnc
        self.fault_generator_fnc = fault_generator_fnc

        self.random_walk_std = fault_random_walk_std

        self.is_train = is_train


        self.observation_space = spaces.Dict({
            'state_estimate': spaces.Box(low = -np.inf, high = np.inf, shape = (int(self.system.state_dim + 0.5 * self.system.state_dim * (self.system.state_dim + 1)),)),  # upper triangular matrix has n(n+1)/2 elements
            'fault_estimate': spaces.Box(low = -np.inf, high = np.inf, shape = (int(self.system.input_dim +  0.5 * self.system.input_dim * (self.system.input_dim + 1)),)),  # upper triangular matrix has n(n+1)/2 elements
            'reference': spaces.Box(low = -np.inf, high = np.inf, shape = (self.system.output_dim,)),
            'system_output': spaces.Box(low = -np.inf, high = np.inf, shape = (self.system.output_dim,)),
            })
        
        
        self.env_obs_shape = (int(self.system.state_dim + 0.5 * self.system.state_dim * (self.system.state_dim + 1) + self.system.input_dim + 0.5 * self.system.input_dim * (self.system.input_dim + 1) + 2 * self.system.output_dim),)
        self.action_space = spaces.Box(low = self.system.min_input, high = self.system.max_input, shape = (self.system.input_dim,))

        self.render_mode = render_mode

    
    def _get_obs(self): 
        return {
            'state_estimate': [self.fault_observer.state_estimate.mean, upper_trinagular(self.fault_observer.state_estimate.cov)],
            'fault_estimate': [self.fault_observer.fault_estimate.mean, upper_trinagular(self.fault_observer.fault_estimate.cov)],
            'system_output': self.system.output, 
            'reference': self.reference[self.step_counter],
        }
    

    def reset(self, seed = None, initial_fault = None, initial_state = None, observer_logger = None):
        '''
        Reset the environment to initial_fault and initial_state if not None, otherwise from the given function generator. 
        
        Args: 
            seed (int): seed for the environment
            initial_fault (np.ndarray): initial fault
            initial_state (np.ndarray): initial state
            observer_logger (object): logger for the observer
        
        Returns: 
            np.ndarray: observation
            dict: info
        ''' 
        super().reset(seed=seed)

        self.step_counter = 0

        initial_state = self.initial_condition_fnc() if (initial_state is None or self.is_train) else initial_state
        initial_fault = self.fault_generator_fnc() if (initial_fault is None or self.is_train) else initial_fault
        self.system.reset(initial_state.reshape((-1,1)))
        self.system.set_fault(initial_fault)
        self.fault_observer.reset()
        if observer_logger is not None: 
            self.obs_logger = deepcopy(observer_logger)
            self.obs_logger.log(self.system, self.fault_observer, None)

        return flatten_and_extract_numbers(self._get_obs()), self._get_info()
    

    def step(self, action): 
        '''
        Take a step in the environment by performing step in system and updating fault obaserver.
        
        Args: 
            action (np.ndarray): action 
            state_noise (np.ndarray): state noise. Default to None, meaning randomly generated noise
            output_noise (np.ndarray): output noise. Default to None, meaning randomly generated noise

        Returns: 
            tuple: observation, reward, done, info
        '''
        self.system.step(action.reshape((-1,1)))
        self.fault_observer.update (A = self.system.A, B=self.system.B, control = action.reshape((-1,1)), state_noise_cov = np.eye(self.system.state_dim) * self.system.state_noise_std**2, 
                                    C = self.system.C, output = self.system.output, output_noise_cov = np.eye(self.system.output_dim) * self.system.output_noise_std**2, fault_random_walk = self.random_walk_std) # controllare questo ultimo passo perchè è definito 

    
        observation = flatten_and_extract_numbers(self._get_obs())  
        reward = self._get_reward()
        info = self._get_info()

        self.step_counter += 1
        return observation, reward, False, False, info  #terminated and truncated are both False
        

    def _get_info(self): 
        return {'cost': self._get_cost(), 
                'step_counter': self.step_counter,
                'dict_state': self._get_obs()}

    def render(self, mode = None): 
        pass
    
    def close(self): 
        pass

    def _get_reward(self): 
        return negative_expected_error_true(self.system.fault.reshape((-1,1)), self.fault_observer.fault_estimate.mean, self.fault_observer.fault_estimate.cov)

    def _get_cost(self): 
        '''
        Binary cost equal to 1 if the l_infty norm of the tracking error is greater than the threshold, 0 otherwise.
        The l_infty norm of the tracking error is the maximum absolute value of the difference between the system output and the reference signal.
        '''
        return 1 if np.linalg.norm(self.system.output-self.reference[self.step_counter], ord=np.inf) > self.track_threshold else 0

    def __getstate__(self): 
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):        
        self.__dict__.update(state)
