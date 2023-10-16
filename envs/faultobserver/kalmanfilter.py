import numpy as np

class GaussianEstimate: 
    def __init__(self, mean, cov): 
        self.mean, self.cov = mean, cov

    @property
    def dim(self): 
        return self.mean.shape[0]


class KalmanFilter(GaussianEstimate): 
    def __init__(self, initial_estimate: GaussianEstimate) -> None: 
        self.mean = initial_estimate.mean
        self.cov = initial_estimate.cov
    
    @property
    def current_estimate(self) -> GaussianEstimate:
        return GaussianEstimate(self.mean, self.cov)

    def update_estimate(
            self, 
            measurement_matrix: np.ndarray,
            measurement_mean: np.ndarray,
            measurement_covariance: np.ndarray
            ) -> None:
        """
        Update the estimate of the state given the measurement.
        C @ x ~ N(mu, Sigma)
        """
        C = measurement_matrix
        K = self.cov @ C.T @ np.linalg.inv(C @ self.cov @ C.T + measurement_covariance)

        self.mean = self.mean + K @ (measurement_mean - C @ self.mean)
        self.cov = (np.eye(self.dim) - K @ C) @ self.cov
    
    def transform_estimate(
        self,
        transform_matrix: np.ndarray,
        noise_mean: np.ndarray,
        noise_cov: np.ndarray,
    ) -> None:
        """
        Transform the estimate of the state given the transformation.
        A @ x + w ~ N(mu, Sigma)
        """
        A = transform_matrix
        self.mean = A @ self.mean + noise_mean
        self.cov = A @ self.cov @ A.T + noise_cov