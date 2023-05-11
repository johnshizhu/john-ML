from keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('X_train: ' + str(x_train.shape))
print('Y_train: ' + str(y_train.shape))
print('X_test:  '  + str(x_test.shape))
print('Y_test:  '  + str(y_test.shape))

# Normalizing data range to 0-1 from 0-255
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_train, axis=1)

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

test_loss, test_acc = model.evaluate(x=x_train, y=y_train)

print('\nTest accuracy:', test_acc)

predictions = model.predict([x_test]) # Make prediction
