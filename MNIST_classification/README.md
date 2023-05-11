# MNIST Multi-Class Classification
NN Classifier using MNIST Handwritten Digit Dataset

### Shape:
X_train --> (60000, 28, 28)<br>
Y_train --> (60000,)<br>
X_test  --> (10000, 28, 28)<br>
Y_test  --> (10000,)<br>

### Normlization
```python
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_train, axis=1)
```

### Model + Training
```python
model = Sequential(
    [
        Flatten(),
        Dense(units=784, activation='relu'),
        Dense(units=128, activation='relu'),
        Dense(units=10,  activation='softmax')
    ]
)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=5)
```
### Ouput Accuracy
```python
test_loss, test_acc = model.evaluate(x=x_train, y=y_train)

print('\nTest accuracy:', test_acc)
```
Test accuracy: 0.993233323097229
