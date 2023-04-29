# Forward Propogation
Implementation of forward propogation from scratch and using Tensorflow
### Compute Dense Layer
Layer functionality with a_in as inputs and W and b as parameters
<br>

```python
def dense(a_in, W, b):
    '''
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    '''
    units = W.shape[1]      # Column count
    a_out = np.zeros(units) # Array of zeros
    for j in range(units):  
        w = W[:,j]
        z = np.dot(w, a_in) + b[j]
        a_out[j] = sigmoid(z)
    return a_out
```
### Example Network with two layers
```
def sequential(x, W1, b1, W2, b2):
    a1 = dense(x,  W1, b1)
    a2 = dense(a1, W2, b2)
    return(a2)
```
### Prediction Function
```
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):      # For every example
        p[i,0] = sequential(X[i], W1, b1, W2, b2)
    return(p)
```
