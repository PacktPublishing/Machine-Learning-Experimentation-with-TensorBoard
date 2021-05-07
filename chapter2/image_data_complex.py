import tensorflow as tf
from tensorflow import keras
import numpy as np
from datetime import datetime
import io
import itertools
from packaging import version
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics


mnist = keras.datasets.mnist
(training_images, training_labels), (val_images, val_labels) = mnist.load_data()

training_images = training_images / 255.0
val_images = val_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu', name='dense_first'),
    tf.keras.layers.Dense(10)
])
 
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
 

logdir = "logs/images/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
file_writer_custom = tf.summary.create_file_writer(logdir)


def log_model_weights(epoch):
    with file_writer_custom.as_default():
        print("epoch_finished")
        tf.summary.image("Weights of 1st Dense Layer", np.reshape(model.get_layer('dense_first').get_weights()[0], (1, 784,128,1)), step=epoch)
 
custom_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_model_weights)


model.fit(
    training_images,
    training_labels,
    epochs=3000,
    callbacks=[tensorboard_callback, custom_callback],
    validation_data=(val_images, val_labels),
)


