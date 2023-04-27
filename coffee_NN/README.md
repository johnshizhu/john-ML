# Tensorflow implementation of a NN
Layer 1 --> 3 neurons, sigmoid
<br>
Layer 2 --> 1 neuron, sigmoid

```python
model = Sequential(
    [
        tf.keras.Input(shape=(2,)), # specifies input shape, typically use model.fit
        tf.keras.layers.Dense(3, activation='sigmoid', name = 'layer1'),
        tf.keras.layers.Dense(1, activation='sigmoid', name = 'layer2')   # typically including sigmoid in final layer is not best practice
    ]
)
```
## model.summary() output:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 layer1 (Dense)              (None, 3)                 9         
                                                                 
 layer2 (Dense)              (None, 1)                 4         
                                                                 
=================================================================
Total params: 13
Trainable params: 13
Non-trainable params: 0
_________________________________________________________________
```

## Testing Trained Model
```
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
print("predictions = \n", predictions)
```
### Predictions
```
predictions = 
 [[9.8621833e-01]
 [8.0845901e-08]]
```
### Decisions
```
decisions = 
[[1.]
 [0.]]
```
