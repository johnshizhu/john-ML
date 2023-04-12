from mpl_toolkits import mplot3d
import copy, math
import numpy as np
import matplotlib.pyplot as plt

# no scientific notation
np.set_printoptions(suppress=True)

def predict_single_loop(x, w, b):
    """
    single predict using linear regression
    Args:
        x (ndarray): Shape (n,) example with multiple features
        w (ndarray): Shape (n,) model parameters    
        b (scalar):  model parameter
        args

    Returns:
      p (scalar):  prediction
    """
    n = x.shape[0]
    p = 0
    for i in range(n):
        p_i = x[i] * w[i]
        p = p + p_i
    p = p + b
    return p

def predict(x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    p = np.dot(x, w) + b
    return p

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
        # print(f"X[i] = {X[i]}")
        # print(f"w = {w}")
        # print(f"b = {b}")
        # print(f"y[i] = {y[i]}")
        # print("BREAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAK")
        err = (np.dot(X[i],w) + b) - y[i]     # calculating the err --> prediction - actual
        for j in range(n):
          # print(f"dj_dw = {dj_dw}")
          # print(f"err = {err}")
          # print(f"j = {j}")
          # print(f"X[i,j] = {X[i,j]}")
          dj_dw[j] = dj_dw[j] + err * X[i, j]  # n is the number of features, updating for change in w
        dj_db = dj_db + err                   # updating the error for change in b

    return dj_db, dj_dw                        #dj_dw returns an dnarray of gradients, while dj_db only returns its own gradient

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

# Load all training data
filename = 'data/ex1data2.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
X_train = data[:, [0,1]]        # House Sizes and number of bedrooms
y_train = data[:, 2]            # House Prices

#print(X_train)

# Initialize values
b_init = 0
w_init = 0

# initialize parameters
initial_w = np.zeros_like(X_train[0])
initial_b = 0

# print("all parameters in order")
# print(f"X_train {X_train}")
# print(f"y_train {y_train}")
# print(f"initial_w {initial_w}")
# print(f"initial_b {initial_b}")

# Settings
iterations = 10000
alpha = 0.0000000001

# Gradient descent
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
print(f"b,w found by gradient descent: {b_final},{w_final} ")



