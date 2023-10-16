import numpy as np

def flatten_dictionary(d:dict) -> np.ndarray: 
    """
    Get all the values of a dictionary and return them as a row vector

    Args: 
        d (dict):   dictionary to flatten.
    
    Returns:
        np.ndarray:    flattened dictionary values.
    """
    values_list = []

    def flatten(value):
        if isinstance(value, np.ndarray):
            if value.ndim > 1:
                value = value.flatten()
            else:
                value = value.tolist()
        return value

    for value in d.values():
        if isinstance(value, tuple):
            for item in value:
                item = flatten(item)
                values_list.extend(item)
        else:
            value = flatten(value)
            values_list.extend(value)

    return np.array(values_list)

def negative_expected_error_true(x:np.ndarray, mu:np.ndarray, cov:np.ndarray) -> float:
    """
    Return the -E[||X-x||^2] where X is a gaussian random variable with covariance matrix 
    cov and mean mu and x is a realization of X

    Args: 
        x (np.ndarray):     vector of shape (1,n) from which compute the error
        mu (np.ndarray):    vector of shape (1,n) representing the mean of the gaussian random variable
        cov (np.ndarray):   matrix of shape (n,n) nonsingular, representing the covariance matrix of the gaussian random variable
    
    Returns: 
        float
    """ 
    x_minus_mu = x - mu
    return -(np.matrix.trace(cov) + x_minus_mu.reshape((1,-1)) @ x_minus_mu)

def sample_uniformly_from_ball(center:np.ndarray, radius:float, size:int) -> np.ndarray:
    """
    Sample a point from a uniform distribution in the ball of radius radius centered in center
    
    Args:
        center (np.ndarray):    center of the ball 
        radius (float):         radius of the ball
        size (int):             size of the point
    
    Returns: 
        np.ndarray: point sampled uniformly in the ball
    """
    d = np.random.uniform(0, radius)
    direction = np.random.normal(0, radius, size=(size,1))
    direction = direction / np.linalg.norm(direction)
    return center + d * direction
