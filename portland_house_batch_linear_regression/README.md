# Linear Regression using the Portland housing prices dataset
This is the code I used to find a "line of best fit" for the classic portland housing price dataset using batch linear regression
<br>

## Cost Function
This cost function returns a value representative of the "accuracy" of the current prediction of housing price. <br>
Args:       <br>
x (ndarray (m,)) -->  Data, m examples <br>
y (ndarray (m,)) -->  Target values    <br>
w,b (scalar)    --> model parameters 

```
def compute_cost(x,y,w,b):
    m = x.shape[0] 
    cost = 0
    for i in range(m):
        f_wb = w*x[i] + b
        cost = cost+(f_wb - y[i])**2
    total_cost = 1/(2*m)*cost
    return total_cost
```

## Gradient Descent
This function computes gradient descent for batch lienar regression <br>
    Args:<br>
        x (ndarray (m,)): Data, m examples <br>
        y (ndarray (m,)): target values    <br>
        w,b (scalar)    : model parameters <br>
```
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
```

## Perform Gradient Descent
This function performs gradient descent to fit w,b.
```
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
```
