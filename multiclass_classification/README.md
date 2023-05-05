# Multi-Class Classification

```python
model = Sequential(
    [
        Dense(units=2, activation='relu', name="L1"),
        Dense(units=4, activation='linear', name="L2")
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
)
model.fit(X_train,y_train,epochs=200)
```
