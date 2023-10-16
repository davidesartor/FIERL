import numpy as np
from scipy.signal import lti 

class LinearStateSystem: 

    def __init__(self, A:np.ndarray, B:np.ndarray, C:np.ndarray, D:np.ndarray, dt:float, sys_type:str, min_input:float = None, max_input:float = None):
        """
        Args: 
            A (np.ndarray): array of shape (n,n) representing the state transition matrix
            B (np.ndarray): array of shape (n,m) representing the input matrix
            C (np.ndarray): array of shape (p,n) representing the output matrix
            D (np.ndarray): array of shape (p,m) representing the feedthrough matrix
            dt (float): sampling time
            sys_type (str): either 'continuous' or 'discrete'
        """
        self.dt = dt

        self.min_input = min_input if min_input is not None else -np.inf
        self.max_input = max_input if max_input is not None else np.inf

        if sys_type == 'continuous':
            self.system = lti(A, B, C, D).to_discrete(self.dt)
        elif sys_type == 'discrete':
            self.system = lti(A, B, C, D)
        else:
            raise ValueError('sys_type must be either continuous or discrete')

    def reset(self, x0:np.ndarray): 
        """
        Args:
            x0 (np.ndarray): array of shape (n,1) representing the initial state
        """
        self.state = x0

    @property
    def A(self): 
        return self.system.A
    
    @property
    def B(self):
        return self.system.B
    
    @property
    def C(self):
        return self.system.C
    
    @property
    def D(self):
        return self.system.D
    
    @property
    def state_dim(self): 
        return self.A.shape[0]

    @property
    def input_dim(self):
        return self.B.shape[1]
    
    @property
    def output_dim(self):
        return self.C.shape[0]
    
    @property
    def output(self): 
        return self.C @ self.state
    
    
    def step(self, input:np.ndarray): 
        """
        Args: 
            input (np.ndarray): array of shape (m,1) representing the input vector
        """
        input = np.clip(input, self.min_input, self.max_input)
        self.state = self.A @ self.state + self.B @ input



class FaultyActuatorNoisySystem(LinearStateSystem): 

    def __init__(self, *args, state_noise_std:float, output_noise_std:float, **kwargs): 
        """
        Args: 
            state_noise_std (float): standard deviation of the transition noise
            output_noise_std (float): standard deviation of the output noise
        """
        super().__init__(*args, **kwargs)
        self.state_noise_std = state_noise_std
        self.output_noise_std = output_noise_std
        self.fault = np.ones(self.input_dim)
    
    def reset(self, x0:np.ndarray):
        super().reset(x0)
        self.fault = np.ones(self.input_dim)
    
    def set_fault(self, fault:np.ndarray):
        """
        Args: 
            fault (np.ndarray): array of shape (m,) representing the fault vector
        """
        self.fault = fault
    
    @property
    def G(self):
        return np.diag(self.fault)
    
    @property
    def output(self):
        return super().output + np.random.normal(0, self.output_noise_std, size=(self.output_dim, 1))
    
    def step(self, input:np.ndarray, state_noise:np.ndarray = None):
        """
        Perform a noisy step in the system. 
        
        Args:
            input (np.ndarray): array of shape (m,1) representing the input vector
            state_noise (np.ndarray): array of shape (n,1) representing the state noise. If None, the state noise is sampled from a normal distribution with standard deviation self.state_noise_std
        """
        input = np.clip(input, self.min_input, self.max_input)
        self.state = self.A @ self.state + self.B @  self.G @ input
        if state_noise is None: 
            self.state += np.random.normal(0, self.state_noise_std, size=(self.state_dim, 1))
        else: 
            self.state += state_noise
        



