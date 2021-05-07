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






!rm -rf logs/scalars
logdir = "logs/scalars/"

file_writer = tf.summary.create_file_writer(logdir)

file_writer.set_as_default()

def lr_schedule(epoch_number):
   """
 Returns a custom learning rate that decreases as epochs progress.
 """
 learning_rate = 0.002
  if epoch_number  epoch > 100:
     learning_rate = 0.0002
 if epoch > 20:

     learning_rate = 0.01
 if epoch > 50:
   learning_rate = 0.005
 tf.summary.scalar('learning rate', data=learning_rate,  step=epoch_number)
   return learning_rate

lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)


tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


model = keras.models.Sequential([
   keras.layers.Dense(16, input_dim=13, activation='relu'),
   keras.layers.Dense(6, activation='relu'),
   keras.layers.Dense(1),
])
model.compile(
   loss='mse', # keras.losses.mean_squared_error
   optimizer=keras.optimizers.SGDAdam(),
)

training_history = model.fit(
   train_features, # input
   train_labels, # output
   batch_size=len(train_labels),
   verbose=0, # Suppress chatty output; use Tensorboard instead
   epochs=1000,
   validation_data=(testval_features, testval_labels),
   callbacks=[tensorboard_callback, lr_callback],
)

