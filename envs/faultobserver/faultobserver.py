import numpy as np
from typing import Union
from copy import deepcopy
from envs.faultobserver.kalmanfilter import KalmanFilter, GaussianEstimate


class FaultObserver: 
    """
    tracks best estimate of system state and faults
    models estimates as gaussian distributions parametrized with mu and cov
    uses bayes rule to update estimates
    """

    def __init__(self, 
                 initial_state_estimate: Union[GaussianEstimate, int],
                 initial_fault_estimate: Union[GaussianEstimate, int],
                 ) -> None: 
        """
        Args:
            initial_state_estimate (Union[GaussianEstimate,int]): 
                if int, it is assumed to be the dimension of the state space
                and the initial state estimate is set to zero and the initial covariance to the identity matrix
                otherwise, it is assumed to be the initial state estimate (column vector) and covariance matrix

            initial_fault_estimate (Union[GaussianEstimate,int])
                if int, it is assumed to be the dimension of the input space
                and the initial fault estimate is set to ones and the initial covariance to the identity matrix
                otherwise, it is assumed to be the initial fault estimate (column vector) and covariance matrix
        """
        # set state estimate 
        if isinstance(initial_state_estimate, int):
            initial_state_estimate = GaussianEstimate(
                np.zeros((initial_state_estimate, 1)),
                np.eye(initial_state_estimate),
            )
        self.state_estimate = KalmanFilter(initial_state_estimate)

        # set fault estimate
        if isinstance(initial_fault_estimate, int):
            initial_fault_estimate = GaussianEstimate(
                0.5 * np.ones((initial_fault_estimate, 1)),
                0.25 * np.eye(initial_fault_estimate),
            )

        self.fault_estimate = KalmanFilter(initial_fault_estimate)

        # save initial estimates
        self.initial_state_estimate = deepcopy(self.state_estimate)
        self.initial_fault_estimate = deepcopy(self.fault_estimate)

    def reset(self, 
              initial_state_estimate: Union[GaussianEstimate, int] = None,
              initial_fault_estimate: Union[GaussianEstimate, int] = None,
              ) -> None:
        """
        reset state and fault estimates to initial values
        
        Args: 
            initial_state_estimate (Union[GaussianEstimate,int])
            initial_fault_estimate (Union[GaussianEstimate,int])
        """
        if initial_state_estimate is None:
            initial_state_estimate = self.initial_state_estimate
        if initial_fault_estimate is None:
            initial_fault_estimate = self.initial_fault_estimate

        self.state_estimate = KalmanFilter(initial_state_estimate)
        self.fault_estimate = KalmanFilter(initial_fault_estimate)
    
    def update(self,
               A: np.ndarray,
               B: np.ndarray,
               control: np.ndarray,
               state_noise_cov: np.ndarray,
               C: np.ndarray,
               output: np.ndarray,
               output_noise_cov: np.ndarray,
               fault_random_walk: float = 0.0,
               ) -> None:
            """
            Update estimates for x and theta.
            Assmunig the following system model:
            x_t+1 ~ N(A * x_t + B * u_t * theta_t, state_noise_cov)
            theta_t+1 ~ N(theta_t, I * fault_random_walk)
            y_t+1 ~ N(C * x_t+1, output_noise_cov)


            Args: 
                A (np.ndarray): 
                    A_t matrix of the system

                B (np.ndarray):
                    B_t matrix of the system

                control (np.ndarray): 
                    u_t control input

                state_noise_cov (np.ndarray): 
                    covariance of the state noise

                C (np.ndarray): 
                    C_t+1 matrix of the system

                output (np.ndarray): 
                    y_t+1 output measurement

                output_noise_cov (np.ndarray): 
                    covariance of the output noise
                    
                fault_random_walk (float, optional):
                    covariance of the fault random walk, by default 0.0
            """

            prior_state_estimate = self.state_estimate.current_estimate
            B_times_u = B @ np.diag(control.flatten())

            # update state estimate to next timestep (a priori)
            # x_t+1  = A * x_t + B * u_t * theta_t + w_t
            self.state_estimate.transform_estimate(
                transform_matrix=A,
                noise_mean=B_times_u @ self.fault_estimate.mean,
                noise_cov=B_times_u @ self.fault_estimate.cov @ B_times_u.T
                + state_noise_cov,
            )

            # update fault estimate to next timestep (a priori)
            # theta_t+1 = theta_t + v_t
            self.fault_estimate.transform_estimate(
                transform_matrix=np.eye(self.fault_estimate.dim),
                noise_mean=np.zeros((self.fault_estimate.dim, 1)),
                noise_cov=fault_random_walk * np.eye(self.fault_estimate.dim),
            )

            # update state estimate with new measurement
            self.state_estimate.update_estimate(
                measurement_matrix=C,
                measurement_mean=output,
                measurement_covariance=output_noise_cov,
            )

            # update fault estimate
            self.fault_estimate.update_estimate(
                measurement_matrix=B_times_u,
                measurement_mean=self.state_estimate.mean - A @ prior_state_estimate.mean,
                measurement_covariance=self.state_estimate.cov
                + A @ prior_state_estimate.cov @ A.T
                + state_noise_cov,
            )


