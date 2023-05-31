import numpy as np

def estimate_gaussian_loops(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """

    m, n = X.shape
    
    mu = np.zeros(n)
    var = np.zeros(n)
    for i in range(n):          # For every feature
        mean_sum = 0
        for j in range(m):      # For every point
            mean_sum += X[j][i]
        mu[i] = mean_sum/m
        var_sum = 0
        for k in range(m):      # For every point
            var_sum += (X[k][i] - mu[i])**2
        var[i] = var_sum/m
            
    return mu, var


def estimate_gaussian(X): 
    """
    Calculates mean and variance of all features 
    in the dataset
    
    Args:
        X (ndarray): (m, n) Data matrix
    
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    
    mu = 1/m * np.sum(X, axis = 0)
    var = 1/m* np.sum((X - mu) ** 2, axis = 0)
            
    return mu, var

def select_threshold(y_val, p_val): 
    """
    Finds the best threshold to use for selecting outliers 
    based on the results from a validation set (p_val) 
    and the ground truth (y_val)
    
    Args:
        y_val (ndarray): Ground truth on validation set
        p_val (ndarray): Results on validation set
        
    Returns:
        epsilon (float): Threshold chosen 
        F1 (float):      F1 score by choosing epsilon as threshold
    """ 

    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        predictions = (p_val < epsilon)
        
        tp = sum((predictions == 1) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        fn = sum((predictions == 0) & (y_val == 1))
        
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)
        
        F1 = (2 * prec * rec)/(prec + rec)
                      
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
        
    return best_epsilon, best_F1
