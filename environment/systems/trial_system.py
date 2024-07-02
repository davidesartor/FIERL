import numpy as np
from environment.systems.system import * 

class ProvaSystem(FaultyActuatorNoisySystem): 
    """
    Implement a linear time-invariant discrete-time system with equations: 

    x_{k+1} = A x_k + B u_k + w_k
    y_k = C x_k + v_k

    where: 
        x_k is the state vector of shape (2,)
        u_k is the input vector of shape (2,)
        y_k is the output vector of shape (1,)
        A is the state matrix of shape (2,2) equal to [[1-alpha, 0], [alpha, 1-beta]]
        B is the input matrix of shape (2,2) equal to [[1, 1], [0, 0]]
        C is the output matrix of shape (1,2) equal to [beta, 0]
        w_k is the state noise vector of shape (2,) iid normal with covariance matrix [[state_noise_std**2, 0], [0, state_noise_std**2]]
        v_k is the output noise vector of shape (1,) iid normal with covariance matrix [[output_noise_std**2]]
    
    Args: 
        alpha: float in [0,1] representing the parameter of the state matrix
        beta: float in [0,1] representing the parameter of the output matrix
        state_noise_std: float representing the standard deviation of the state noise
        output_noise_std: float representing the standard deviation of the output noise
        min_input: float representing the minimum value of the input
        max_input: float representing the maximum value of the input
    """
    def __init__(self, 
                 alpha = 0.5, 
                 beta = 0.5,
                 gamma = 0.5,
                 state_noise_std = 1e-4, 
                 output_noise_std = 1e-3,
                 min_input = -0.002,
                 max_input = 0.02, 
                 number_of_actuator = 2,
                 ):

        # A = np.array([[1-alpha, 0], [alpha, 1-beta]])
        A = np.array([[1-alpha, 0, gamma], [alpha, 1-beta, 0], [0, 0, 1]]) # with constant input equal to 1
        # B = np.array([[1, 1], [0, 0]])
        # B = np.array([[1], [0]])
        # B = np.array([[1, 1, 1], [0, 0, 0]])

        B = np.array([[1]*number_of_actuator, [0]*number_of_actuator, [0]*number_of_actuator])
        # B = np.array([[1, 1], [0, 0], [0, 0]]) if 

        C = np.array([0, beta, 0]) # with constant input equal to 1
        # D = np.array([[0, 0]])
        # D = np.array([[0]])
        D = np.zeros((1, number_of_actuator))
        # D = np.array([np.zeros(number_of_actuator)])
        

        super().__init__(A, B, C, D, dt = 1.0, sys_type = 'discrete', min_input = min_input, max_input = max_input, state_noise_std = state_noise_std, output_noise_std = output_noise_std)
        