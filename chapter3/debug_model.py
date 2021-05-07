!pip install -U tensorboard_plugin_profile
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
from sklearn import model_selection
from shutil import move as mv


!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip \
    -O /tmp/horse-or-human.zip

import os
import zipfile

local_zip = '/tmp/horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp/horse-or-human')
zip_ref.close()

tf.debugging.experimental.enable_dump_debug_info(
        "logs/debugging",
        tensor_debug_mode="FULL_HEALTH",
        circular_buffer_size=-1)

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
tf.keras.layers.MaxPool2D(2,2),
#second convolution
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
#third convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
#fourth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
#fifth convolution
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPool2D(2,2),
#Flatten
tf.keras.layers.Flatten(),
#fully connected layer
tf.keras.layers.Dense(512, activation='relu'),
#output layer
tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
    '/tmp/horse-or-human/',
    target_size = (300,300),
    batch_size = 128,
    class_mode = 'binary'
)

model.fit(train_generator,
          epochs=2,
          validation_data=train_generator
)


