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

logdir = "logs/images/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
file_writer = tf.summary.create_file_writer(logdir)


training_images = training_images / 255.0
val_images = val_images / 255.0

with file_writer.as_default():
  images = np.reshape(training_images[:50], (-1, 28, 28, 1))
  tf.summary.image("Points", images, max_outputs=100, step=0)




