import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import keras.api._v2.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# make_blobs to make a training dataset
# making a 4 class dataset
classes = 4
m = 100
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
std = 1.0
X_train, y_train = make_blobs(n_samples=m, centers=centers, cluster_std=std,random_state=30)

tf.random.set_seed(1234)
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