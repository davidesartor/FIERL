from typing import NamedTuple, Union, Tuple
import numpy as np
from copy import deepcopy 

class GaussianEstimate(NamedTuple): 
    mean: np.ndarray
    cov: np.ndarray

class KalmanFilter:

    def __init__(self, initial_estimate:GaussianEstimate): 
        self.estimate = deepcopy(initial_estimate)

    def reset(self, initial_estimate:GaussianEstimate): 
        self.estimate = deepcopy(initial_estimate)
       
    def update_a_priori(self, A:np.ndarray, B:np.ndarray, noise_cov_x:Union[np.ndarray, int, float], u:np.ndarray): 
        if isinstance(noise_cov_x, (int, float)): 
            noise_cov_x = noise_cov_x * np.eye(len(self.estimate.mean))
        self.estimate = GaussianEstimate(
            mean = A @ self.estimate.mean + B @ u, 
            cov = A @ self.estimate.cov @ A.T + noise_cov_x
        )
    
    def update_a_posteriori(self, C:np.ndarray, noise_cov_y:Union[np.ndarray, int, float], y:np.ndarray): 
        if isinstance(noise_cov_y, (int, float)): 
            noise_cov_y = noise_cov_y * np.eye(len(y))
        K = self.estimate.cov @ C.T @ np.linalg.inv(C @ self.estimate.cov @ C.T + noise_cov_y)
        self.estimate = GaussianEstimate(
            mean = self.estimate.mean + K @ (y - C @ self.estimate.mean), 
            cov = (np.eye(self.estimate.cov.shape[0]) - K @ C) @ self.estimate.cov
        )
    
    def get_estimate(self): 
        return self.estimate

class FaultObserver(KalmanFilter): 
    
    def __init__(self, state_dim:int, input_dim:int, fault_evol_cov:Union[np.ndarray, float, int], initial_state_estimate:GaussianEstimate = None, initial_fault_estimate:GaussianEstimate = None): 
        '''
        Args: 
            state_dim (int): dimension of the state
            input_dim (int): dimension of the input
            initial_state_estimate (GaussianEstimate, Optional): initial estimate of the state, if None, it is set to zero mean and identity covariance
            initial_fault_estimate (GaussianEstimate, Optional): initial estimate of the fault, if None it is set to ones mean and identity covariance
        '''
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.fault_evol_cov = fault_evol_cov
        self.estimate = self.reset(initial_state_estimate, initial_fault_estimate)

    def reset(self, initial_state_estimate:GaussianEstimate = None, initial_fault_estimate:GaussianEstimate = None): 
        '''
        Reset the fault observer to the initial state and fault estimates
        '''
        if initial_state_estimate is None: 
            initial_state_estimate = GaussianEstimate(
                mean = np.zeros((self.state_dim,1)), 
                cov = np.eye(self.state_dim)
            )
        if initial_fault_estimate is None:
            initial_fault_estimate = GaussianEstimate(
                mean = 0.5 * np.ones((self.input_dim,1)), 
                cov = np.eye(self.input_dim)
            )
        zeros = np.zeros((self.state_dim, self.input_dim)) # state_dim x fault_dim matrix 
        self.estimate = GaussianEstimate(
            mean = np.concatenate((initial_state_estimate.mean, initial_fault_estimate.mean)),
            cov = np.block([[initial_state_estimate.cov, zeros], [zeros.T, initial_fault_estimate.cov]])
        )
        return self.estimate
    
    def update(self, y:np.ndarray, C:np.ndarray, output_noise_cov: Union[np.ndarray, int, float], 
               u:np.ndarray = None, A:np.ndarray = None, B:np.ndarray = None, state_noise_cov:Union[np.ndarray, int, float] = None): 
        '''
        Create augmented state with x and fault and update the fault observer with the control input and the output measurement. 
        Args: 
            y (np.ndarray): measurement vector
            C (np.ndarray): output matrix
            output_noise_cov (Union[np.ndarray, int, float]): output noise covariance
            u (np.ndarray): control input
            A (np.ndarray): state transition matrix
            B (np.ndarray): input matrix
            state_noise_cov (Union[np.ndarray, int, float]): state noise covariance
        '''
        C_bar = [C, np.zeros((C.shape[0], self.input_dim))]
        
        if A is not None and B is not None and state_noise_cov is not None and u is not None: 
            A_bar = [[A, B @ np.diag(u.flatten())], [np.zeros_like(B.T), np.eye(self.input_dim)]]
            B_bar = np.zeros((self.state_dim + self.input_dim, self.input_dim))
            zeros = np.zeros((self.state_dim, self.input_dim))
            noise_cov_x_bar = np.block([[state_noise_cov, zeros], [zeros.T, self.fault_evol_cov * np.eye(self.input_dim)]])
            self.update_a_priori(np.block(A_bar), B_bar, noise_cov_x_bar, u)
            self.update_a_posteriori(np.block(C_bar), output_noise_cov, y)
            
        else: 
            self.update_a_posteriori(np.block(C_bar), output_noise_cov, y)
            print('Updating only with measurement.')
        
        return self.estimate

    def split(self) -> Tuple[GaussianEstimate, GaussianEstimate]: 
        state_estimate = GaussianEstimate(
            mean = self.estimate.mean[:self.state_dim],
            cov = self.estimate.cov[:self.state_dim, :self.state_dim]
        )
        fault_estimate = GaussianEstimate(
            mean = self.estimate.mean[self.state_dim:],
            cov = self.estimate.cov[self.state_dim:, self.state_dim:]
        )
        return state_estimate, fault_estimate

        
