from typing import Any
import numpy as np

def flatten_and_extract_numbers(data):
    '''
    Pick all the numbers (int or float) from a nested list, dictionary or numpy array 
    and return them in a single list. 
    '''
    numbers = []

    if isinstance(data, dict):
        for value in data.values():
            numbers.extend(flatten_and_extract_numbers(value))
    elif isinstance(data, list):
        for item in data:
            numbers.extend(flatten_and_extract_numbers(item))
    elif isinstance(data, np.ndarray):
        if data.size == 1:  # if it's a single element array
            numbers.append(data.item())  # extract the single element
        else:
            numbers.extend(flatten_and_extract_numbers(data.tolist()))  # flatten the array
    elif isinstance(data, (int, float)):
        numbers.append(data)

    return np.array(numbers)


def upper_trinagular(matrix): 
    '''
    Return a flattened array of the upper triangular part of the matrix.
    '''
    upper_indices = np.triu_indices(matrix.shape[0])
    return matrix[upper_indices]

# REWARD FUNCTIONS 
def negative_expected_error_true(x, mean, covariance): 
    '''
    Return -E[||X - x||**2] where X is a gaussian random variable with covariance matrix covariance and mean mean, x is a realization of x.
    
    Args: 
        x (np.ndarray): array of shape (n,) representing the realization of the random variable
        mean (np.ndarray): array of shape (n,) representing the mean of the random variable
        covariance (np.ndarray): array of shape (n,n) representing the covariance matrix of the random variable
    
    Returns: 
        float: -E[||X - x||**2]
    '''

    x_minus_mu = x - mean
    r = -(np.matrix.trace(covariance) + x_minus_mu.reshape((1, -1)) @ x_minus_mu)
    return np.squeeze(r)



# INITIAL CONDITION AND FAULT SAMPLING FUNCTIONS

def sampling_uniformly_from_n_ball_rejection(r:float, c:np.ndarray, n_samples:int): 
    '''
    Return n_samples uniformly distributed in the n-ball with Rejection Sampling.

    Args: 
        r (float): radius of the sphere
        c (np.ndarray): array of shape (d,) representing the center of the sphere
        n_samples (int): number of samples

    Returns:
        np.ndarray: array of shape (n_samples, d) representing the samples
    '''
    points = []
    d = c.shape[0]
    while len(points) < n_samples: 
        x = np.random.rand(d) * 2 * r - r + c
        if np.linalg.norm(x - c) < r: 
            x = np.array(x)
            points.append(x)
    points = np.array(points)  
    return points


def sampling_uniformly_from_n_ball_muller(r:float, c:np.ndarray, n_samples:int): 
    '''
    Return n_samples uniformly distributed in the n-ball with Muller's method.
    
    Args: 
        r (float): radius of the sphere
        c (np.ndarray): array of shape (d,) representing the center of the sphere
        n_samples (int): number of samples
    
    Returns:
        np.ndarray: array of shape (n_samples, d) representing the samples
    '''
    points = []
    d = c.shape[0]
    while len(points) < n_samples: 
        u = np.random.normal(0, 1, d).reshape(-1,1) # an array of normally distributed random variables
        norm = np.linalg.norm(u)
        radius = np.random.rand() ** (1/d) * r
        x = c + radius * u / norm
        x = np.array(x)
        points.append(x)
    points = np.array(points)
    # if only one sample is requested, return a single array
    if n_samples == 1:
        return points[0]
    return points


def sampling_uniformly_from_n_cube(half_side:float, center:np.ndarray, n_samples:int): 
    '''
    Return n_samples uniformly distributed in the n-cube.
    
    Args: 
        half_side (float): half side of the cube
        center (np.ndarray): array of shape (d,) representing the center of the cube
        n_samples (int): number of samples
    '''

    d = center.shape[0]
    return np.random.rand(n_samples, d) * 2 * half_side - half_side + center


def sampling_uniformly_fault(a, b, size): 
    return np.random.uniform(a, b, size = size) # controllare in caso, si usava random.uniform sulla versione di TF1. 



class InitialConditionSampler:
    '''
    This class is needed to serialize the Environment class.
    '''
    def __init__(self, ic_type, **kwargs):
        self.ic_type = ic_type
        self.kwargs = kwargs
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.ic_type in ['cube', 'square', 'hypercube', 'ncube']: 
            return sampling_uniformly_from_n_cube(**self.kwargs)
        elif self.ic_type in ['ball', 'nball', 'circle']:
            return sampling_uniformly_from_n_ball_muller(**self.kwargs)
        elif self.ic_type in ['gaussian', 'normal']:
            return np.random.normal(**self.kwargs)
        else: 
            raise ValueError(f'Unknown type {self.ic_type}')

class FaultSampler: 
    '''
    This class is needed to serialize the Environment class.
    '''
    def __init__(self, fault_type, **kwargs):
        self.fault_type = fault_type
        self.kwargs = kwargs
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.fault_type in ['uniform']:
            return sampling_uniformly_fault(**self.kwargs)
        else:
            raise ValueError(f'Unknown type {self.fault_type}')
     
