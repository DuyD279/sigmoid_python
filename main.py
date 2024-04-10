import pandas as pd

dataset = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

x = dataset.drop(columns=["Age", "Diameter"])

y = dataset["Length"]

"""Split the data into a training set and a testing set."""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

"""Build and train the model."""

import tensorflow as tf

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(256, input_shape=x_train.shape[1:], activation='sigmoid'))
model.add(tf.keras.layers.Dense(256, activation='sigmoid'))
model.add(tf.keras.layers.Dense(1, activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

model.fit(x_train, y_train, epochs=100)

"""Evaluate the model."""

model.evaluate(x_test, y_test)