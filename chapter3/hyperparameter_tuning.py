import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
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

dropout_hp = hp.HParam('dropout', hp.RealInterval(0.5, 0.7))
opimizer_hp = hp.HParam('optimizer', hp.Discrete(['sgd', 'adam']))

metric = 'accuracy'

with tf.summary.create_file_writer('logs/hp_tuning').as_default():
  hp.hparams_config(
    hparams=[dropout_hp, opimizer_hp],
    metrics=[hp.Metric(metric, display_name='Accuracy')],
  )


def hp_model(hparams):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(hparams[dropout_hp]),
    #second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(hparams[dropout_hp]),
    #third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(hparams[dropout_hp]),
    #fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(hparams[dropout_hp]),
    #fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Dropout(hparams[dropout_hp]),
    #Flatten
    tf.keras.layers.Flatten(),
    #fully connected layer
    tf.keras.layers.Dense(512, activation='relu'),
    #output layer
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
      optimizer=hparams[opimizer_hp],
      loss='binary_crossentropy',
      metrics=['accuracy']
    )

    model.fit(
        ds_train,
        epochs = 1
    )
    _, acc = model.evaluate(ds_test)
    print(acc)
    return acc

def run_hp(dir, hparams):
    with tf.summary.create_file_writer(dir).as_default():
        hp.hparams(hparams)
        acc = hp_model(hparams)
        print(metric)
        tf.summary.scalar(metric, acc, step=1)


sess = 0
for dropout in (dropout_hp.domain.min_value, dropout_hp.domain.max_value):
    for optimizer in opimizer_hp.domain.values:
        hparams = {
            dropout_hp: dropout,
            opimizer_hp: optimizer
        }
        run_hp('logs/hp_tuning/' + str(sess), hparams)
        sess = sess + 1
