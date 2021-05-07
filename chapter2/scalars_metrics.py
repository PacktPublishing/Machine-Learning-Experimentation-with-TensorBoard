import tensorflow as tf
from datetime import datetime
from packaging import version
from tensorflow import keras
import numpy as np

def celsius_to_fahrenheit(c):
    return (c * (9/5)) + 32


 
celsius_points = list(range(-1000, 1000))
fahrenheit_points = [celsius_to_fahrenheit(c) for c in celsius_points]
val_features = celsius_points[:50] + celsius_points[-50:]
val_labels = fahrenheit_points[:50] + fahrenheit_points[-50:]
 
train_features = celsius_points[50:] + celsius_points[:-50]
train_labels = fahrenheit_points[50:] + fahrenheit_points[:-50]


logdir = "logs/scalars/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1, activation='relu'),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1)
])


model.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.Adam(lr=1e-3),
)



model.fit(
    train_features, # input
    train_labels, # output
    batch_size=len(train_labels),
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=1000,
    validation_data=(val_features, val_labels),
    callbacks=[tensorboard_callback],
)



