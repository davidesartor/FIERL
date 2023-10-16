import numpy as np
import random
from copy import deepcopy

def print_statistics(title, data, num_digits=5):
    print(title)
    print("\tmean: ", np.round(np.mean(data), num_digits))
    print("\tstd: ", np.round(np.std(data), num_digits))
    print("\tmedian: ", np.round(np.median(data), num_digits))
    print("\tmin: ", np.round(np.min(data), num_digits))
    print("\tmax: ", np.round(np.max(data), num_digits))
    print("\t25perc: ", np.round(np.percentile(data, 25), num_digits))
    print("\t75perc: ", np.round(np.percentile(data, 75), num_digits))
    print()

def save_statistics_to_file(filename, title, data, num_digits=5):
    with open(filename, 'a') as f:
        f.write(title + '\n')
        f.write("\tmean: " + str(np.round(np.mean(data), num_digits)) + '\n')
        f.write("\tstd: " + str(np.round(np.std(data), num_digits)) + '\n')
        f.write("\tmedian: " + str(np.round(np.median(data), num_digits)) + '\n')
        f.write("\tmin: " + str(np.round(np.min(data), num_digits)) + '\n')
        f.write("\tmax: " + str(np.round(np.max(data), num_digits)) + '\n')
        f.write("\t25perc: " + str(np.round(np.percentile(data, 25), num_digits)) + '\n')
        f.write("\t75perc: " + str(np.round(np.percentile(data, 75), num_digits)) + '\n\n')

def adjust_list_length(arr:list, desired_length:int) -> list:
    """
    Adjust the length of a to the desired length. 
    If the desired length is larger than the original length, the array is repeated until the desired length is reached.
    If the desired length is smaller than the original length, the array is truncated.
    Args: 
        arr (list): list to be adjusted
        desired_length (int): desired length of the list
    Returns:
        list: adjusted list
    """
    original_length = len(arr)
    new_arr = deepcopy(arr)
    
    if desired_length >= original_length:
        while len(new_arr) < desired_length:
            new_arr.append(arr[len(new_arr) % original_length])
    else:
        new_arr = new_arr[:desired_length]
    return new_arr


def ordered_constrained_list(max_value:int, min_diff:int, max_diff:int, adaptive:bool = False) -> list: 
    """
    Generate a ordered list with elements from minsteps to max_value, where between two subsequent
    elements the difference is at least min_diff and at most max_diff. 

    Args: 
        max_value (int): maximum value of the list
        min_diff (int): minimum gap between two subsequent elements
        max_diff (int): maximum gap two subsequent elements
        adaptive (bool, optional): if False, maxvalue is strictly enforced, if True, the last element > max_value is allowed
    
    Returns: 
        lst: ordered list of integers
    """
    if min_diff<0 or max_diff<0 or min_diff>max_diff:
        raise ValueError('min_diff and max_diff must be positive and min_diff <= max_diff')
    
    lst = []
    current_value = 0 

    while current_value <= max_value:
        current_value += random.randint(min_diff, max_diff)
    
        if len(lst) != 0: 
            if current_value != lst[-1]:
                lst.append(current_value)
        else:
            lst.append(current_value)
    
    if adaptive: 
        return lst
    
    if len(lst)<=1: 
        return list([max_value+1])

    return lst[:-1]


def ordered_constrained_fixedlen_list(max_value, min_diff, max_diff, max_len): 
    if max_len * max_diff >= max_value - 1: 
        raise ValueError("Impossible to generate a list of length {} with max_value {}, min_diff {} and max_diff {}".format(max_len, max_value, min_diff, max_diff))
    
    lst = []
    current_value = 0 
    while len(lst) < max_len: 
        current_value = current_value + random.randint(min_diff, max_diff)
        if current_value < max_value: 
            lst.append(current_value)
    return lst

