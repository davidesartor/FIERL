import numpy as np
from copy import deepcopy
import gymnasium as gym 
from gymnasium import spaces 
import inspect
from environment.utils import * 
import inspect
from environment.faultobserver.faultobserver import GaussianEstimate


class Environment(gym.Env): 
    
    def __init__(self, system, fault_observer, reference, track_threshold, initial_condition_fnc, fault_generator_fnc, max_ep_len, render_mode = None, observer_logger = None): 
        '''
        Args: 
            system (object): system to be controlled
            fault_observer (object): fault observer
            reference (np.ndarray): reference signal to track
            track_threshold (float): threshold on tracking error
            initial_condition_fnc (callable): function to generate initial conditions of the system 
            fault_generator_fnc (callable): function to generate faults
            max_ep_len (int): maximum episode length
            render_mode (str, optional): render mode
            observer_logger (object, optional): logger for the observer
        '''

        super().__init__()
      
        self.system = deepcopy(system)
        self.fault_observer = deepcopy(fault_observer)
        self.observer_logger = deepcopy(observer_logger)
        self.reference = reference
        self.track_threshold = track_threshold
        self.initial_condition_fnc = initial_condition_fnc
        self.fault_generator_fnc = fault_generator_fnc
        self.max_ep_len = max_ep_len
        self.render_mode = render_mode

        if len(self.reference) != self.max_ep_len:
            print('Reference signal length does not match episode length.')

        state_fault_dim = self.system.state_dim + self.system.input_dim
        self.observation_space = spaces.Dict({
            # 'state_estimate': spaces.Box(low = -np.inf, high = np.inf, shape = (int(self.system.state_dim + 0.5 * self.system.state_dim * (self.system.state_dim + 1)),)),  # upper triangular matrix has n(n+1)/2 elements
            # 'fault_estimate': spaces.Box(low = -np.inf, high = np.inf, shape = (int(self.system.input_dim +  0.5 * self.system.input_dim * (self.system.input_dim + 1)),)),  # upper triangular matrix has n(n+1)/2 elements
            'state_fault_estimate': spaces.Box(low = -np.inf, high = np.inf, shape = (int(state_fault_dim + 0.5 * state_fault_dim * (state_fault_dim + 1) ),)),
            'reference': spaces.Box(low = -np.inf, high = np.inf, shape = (self.system.output_dim,)),
            'system_output': spaces.Box(low = -np.inf, high = np.inf, shape = (self.system.output_dim,)),
            })
        self.action_space = spaces.Box(low = self.system.min_input, high = self.system.max_input, shape = (self.system.input_dim,))

    
    def _get_obs(self): 
        # state_estimate, fault_estimate = self.fault_observer.split()
        return {
            # 'state_fault_estimate': [state_estimate.mean, upper_triangular(state_estimate.cov)],
            # 'fault_estimate': [fault_estimate.mean, upper_triangular(fault_estimate.cov)],
            'state_fault_estimate': [self.fault_observer.estimate.mean, upper_triangular(self.fault_observer.estimate.cov)],
            'system_output': self.system.output, 
            'reference': self.reference[self.step_counter],
        }
    

    def reset(self, initial_fault = None, initial_state = None, seed = None):
        '''
        Reset the environment.
        
        Args: 
            initial_fault (np.ndarray): initial fault, shape (system.input_dim, )
            initial_state (np.ndarray): initial state, shape (system.state_dim, )
            seed (int): seed for the environment
        
        Returns: 
            np.ndarray: observation
            dict: info
        ''' 
        super().reset(seed=seed)

        self.step_counter = 0

        initial_state = self.initial_condition_fnc() if initial_state is None else initial_state
        self.system.reset(initial_state.reshape((-1,1)))

        initial_fault = self.fault_generator_fnc() if initial_fault is None else initial_fault
        self.system.set_fault(initial_fault)

        ## CHANGE to start with a better estimate
        ise = GaussianEstimate(initial_state.reshape(-1, 1), cov = 0.1 * np.eye(self.system.state_dim))
        ife = GaussianEstimate(initial_fault.reshape((-1, 1)), cov = 0.1 * np.eye(self.system.input_dim))
   
        self.fault_observer.reset(initial_state_estimate = ise, initial_fault_estimate = ife)
        # ===================================================================

        # self.fault_observer.reset()
        # self.fault_observer.update(y = self.system.output, C = self.system.C, output_noise_cov = np.eye(self.system.output_dim) * self.system.output_noise_std**2,)

        if self.observer_logger is not None: 
            self.observer_logger.reset()
            self.observer_logger.log(system = self.system, observer = self.fault_observer, control_input = None, reference = self.reference[self.step_counter])

        return np.expand_dims(flatten_and_extract_numbers(self._get_obs()), axis=0), self._get_info()
    

    def step(self, action): 
        '''
        Take a step in the environment by performing step in system and updating fault observer.
        
        Args: 
            action (np.ndarray): action 

        Returns: 
            tuple: observation, reward, terminated, truncated, info
        '''
        action = np.clip(action, self.system.min_input, self.system.max_input)
        self.system.step(action.reshape((-1,1)))
        self.fault_observer.update(y = self.system.output, C = self.system.C, output_noise_cov = np.eye(self.system.output_dim) * self.system.output_noise_std**2, 
                                    u = action.reshape((-1,1)), A = self.system.A, B=self.system.B, state_noise_cov = np.eye(self.system.state_dim) * self.system.state_noise_std**2)
        if self.observer_logger is not None: 
            self.observer_logger.log(system = self.system, observer = self.fault_observer, control_input = action.reshape((-1,1)), reference = self.reference[self.step_counter])

        observation = np.expand_dims(flatten_and_extract_numbers(self._get_obs()), axis=0)
        reward = np.expand_dims(self._get_reward(), axis=0)
        info = self._get_info()
        info['cost'] = np.expand_dims(info['cost'], axis=0) 
        terminated = np.expand_dims(False, axis=0)
        truncated = np.expand_dims(False, axis=0)

        self.step_counter += 1
        
        if self.step_counter == self.max_ep_len: # truncate episode
            info['final_observation'] = observation #already expanded
            truncated = np.expand_dims(True, axis=0)
            return observation, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    def _get_info(self): 
        return {'cost': self._get_cost(), 
                'step_counter': self.step_counter,
                'dict_state': self._get_obs()}

    def render(self, mode = None, save:bool = False, save_path:str = None): 
        if self.observer_logger is not None: 
            self.observer_logger.plot(save=save, save_path = save_path, track_threshold= self.track_threshold)
        else: 
            print('No observer logger found.')
    
    def close(self): 
        pass

    def _get_reward(self): 
        _, fault_estimate = self.fault_observer.split()
        # print('reward', negative_expected_error_true(self.system.fault.reshape(-1,1), fault_estimate.mean, fault_estimate.cov).item())
        return negative_expected_error_true(self.system.fault.reshape(-1,1), fault_estimate.mean, fault_estimate.cov).item()

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
