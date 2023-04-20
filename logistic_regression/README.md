# Logistic Regression
Functions:

### Cost Function
```python
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b    # "z" linear function
        f_wb_i = 1/(1+(np.exp(-(z_i))))
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
    cost = cost / m
    return cost
```

Returning the cost

### Computing Logistic Gradient
```python
    m,n = X.shape
    dj_dw = np.zeros((n,))                           
    dj_db = 0.

    for i in range(m):
        f_wb_i = 1/(1+(np.exp(-(np.dot(X[i],w) + b)))) 
        err_i  = f_wb_i  - y[i]                       
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]      
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m                                   
    dj_db = dj_db/m                                   
        
    return dj_db, dj_dw  
```

Returning dj_dw and dj_db

### Performing Logistic Gradient Descent
```python
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):
        # Find the gradient and update the parameters
        dj_db, dj_dw = compute_gradient_logistic(X, y, w, b)

        # Update parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save the cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( compute_cost_logistic(X, y, w, b) )
        
        # Printing at intervals
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history
```
Performs logistic regression and prints out the decreasing cost at specified intervals
<br><br>
Example of a line created by logistic regression for a small dataset, where shading is correlated with certainty of 0 or 1
![LogisticRegExample](https://user-images.githubusercontent.com/115199074/233491243-860cee68-5230-4631-8c06-c87e3d1985ba.png)

