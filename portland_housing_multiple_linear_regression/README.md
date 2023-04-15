# Multiple Linear Regression
This is code for multiple linear regression of the portland housing dataset, considering house size and bedroom count to predict price. 

## Cost Function
Function to compute the "cost" of the current state of the model. A measure of how accurate the model is.
```
def compute_cost(X, y, w, b):
    """
    compute cost function
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter  

    Returns:
      cost (scalar): cost  
    """
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
      f_wb_i = np.dot(X[i], w) + b
      cost = cost + (f_wb_i - y[i])**2
    cost = cost/(2*m)  
    return cost
```

## Gradient Descent
Function to determine the rate of change (partial derivative) of the cost function.
```
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """   
    m,n = X.shape
    dj_dw = np.zeros((n,)) 
    dj_db = 0
    
    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]     # calculating the err --> prediction - actual
        for j in range(n):
          dj_dw[j] = dj_dw[j] + err * X[i, j]  # n is the number of features, updating for change in w
        dj_db = dj_db + err                   # updating the error for change in b

    return dj_db, dj_dw                        #dj_dw returns an dnarray of gradients, while dj_db only returns its own gradient
```

## Perform Gradient Descent
A function to perform gradient descent based on your training data set and specified learning rate and number of iterations.
```
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
  
  # store cost J and ws in array for graphing later
    J_history = []
    w = copy.deepcopy(w_in) # avoid modifying theclobal w within fucntion
    b = b_in

    for i in range (num_iters):
        
        # Find the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update the parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save the cost J at each iteration
        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i}: Cost={J_history[-1]} w={w} b={b} ")
    return w, b, J_history
```
### Note
Learning rate (alpha) can change the efficacy of this function high learning ratge can results in ineffective function, while low learning rate will result in a slow learning process. 
