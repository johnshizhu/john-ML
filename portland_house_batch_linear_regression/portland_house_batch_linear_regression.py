# Linear Regression through gradient descent using the portland housing price dataset. 
import math, copy
import numpy as np
import matplotlib.pyplot as plt

# no scientific notation
np.set_printoptions(suppress=True)

# Cost Function
def compute_cost(x,y,w,b):
    """
    The Cost function
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters
    Returns:
        total_cost      : Cost
    """
    m = x.shape[0] 
    cost = 0
    
    for i in range(m):
        f_wb = w*x[i] + b
        cost = cost+(f_wb - y[i])**2
    total_cost = 1/(2*m)*cost

    return total_cost

# Gradient Descent
def compute_gradient(x,y,w,b):
    """
    Computes gradient descent for linear regression
    Args:
        x (ndarray (m,)): Data, m examples
        y (ndarray (m,)): target values
        w,b (scalar)    : model parameters
    Returns:
        dj_dw (scalar)  : Gradient of the cost with respect to the parameters w
        dj_db (scalar)  : Gradient of the cost with respect to the parameter  b
    """
    # Number of training examples
    m = x.shape[0]    
    dj_dw = 0
    dj_db = 0
    
    for i in range(m):  
        f_wb = w * x[i] + b 
        dj_dw_i = (f_wb - y[i]) * x[i] 
        dj_db_i = f_wb - y[i] 
        dj_db += dj_db_i
        dj_dw += dj_dw_i 
    dj_dw = dj_dw/m 
    dj_db = dj_db/m 
        
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function): 
    """
    Performs gradient descent to fit w,b. Updates w,b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      x (ndarray (m,))  : Data, m examples 
      y (ndarray (m,))  : target values
      w_in,b_in (scalar): initial values of model parameters  
      alpha (float)     : Learning rate
      num_iters (int)   : number of iterations to run gradient descent
      cost_function     : function to call to produce cost
      gradient_function : function to call to produce gradient
      
    Returns:
      w (scalar)        : Updated value of parameter after running gradient descent
      b (scalar)        : Updated value of parameter after running gradient descent
      J_history (List)  : History of cost values
      p_history (list)  : History of parameters [w,b] 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in
    
    for i in range(num_iters):
        # Calculate the new derivatives using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w , b)     

        # Update parameters
        b = b - alpha * dj_db                            
        w = w - alpha * dj_dw                            

        # Saving information for displaying later
        if i<100000:                            # prevent resource exhaustion 
            J_history.append( cost_function(x, y, w , b))
            p_history.append([w,b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history #return w and J,w history for graphing

# Load all training data
filename = 'data/ex1data2.txt'
data = np.loadtxt(filename, delimiter=',', skiprows=1)
x_train = data[:,0]     # House Sizes
y_train = data[:, 2]    # House Prices

# Initialize parameters
w_init = 0
b_init = 0
# Initial gradient descent settings
iterations = 10000
tmp_alpha = 0.0000001
# Run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")

# Display the data with the newly created line of best fit
x = np.linspace(0,5000,100)
y = w_final*x + b_final
plt.scatter(x_train, y_train)
plt.plot(x,y,'-r')
plt.xlabel("House Size (sqft)")
plt.ylabel("House Price")
plt.show()
