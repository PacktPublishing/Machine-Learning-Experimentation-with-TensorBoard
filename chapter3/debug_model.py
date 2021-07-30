import tensorflow as tf
import numpy as np

import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load('HorsesOrHumans', split=['train', 'test'], with_info=True, as_supervised=True, shuffle_files=True)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.cache()
ds_test = ds_test.batch(128)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


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

model.fit(ds_train,
          epochs=2,
          validation_data=ds_test
)


