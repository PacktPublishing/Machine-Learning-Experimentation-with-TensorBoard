import tensorflow as tf
from datetime import datetime
from packaging import version
from tensorflow import keras
import numpy as np

def celsius_to_fahrenheit(c):
    return (c * (9/5)) + 32


model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1, activation='relu'),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1)
])


logdir = "logs/graph/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model.compile(
    optimizer='SGD',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])


random_training_images = np.random.normal(size=(500,224,224,3))
random_training_labels = list(range(0, 500))
random_training_labels[-1] = 999



model.fit(random_training_images,
          random_training_labels,
          epochs=1,
          batch_size=32,
          callbacks=[tensorboard_callback])



