# Anomaly Detection using Gaussian

Python implementation of Anomaly Detection

# Estimate Gaussian
```python
def estimate_gaussian(X): 
    m, n = X.shape
    
    mu = 1/m * np.sum(X, axis = 0)
    var = 1/m* np.sum((X - mu) ** 2, axis = 0)
            
    return mu, var
```

# Select Threshold
```python
def select_threshold(y_val, p_val): 

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

```
