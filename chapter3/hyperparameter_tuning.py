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

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
    train_generator = train_datagen.flow_from_directory(
        '/tmp/horse-or-human/',
        target_size = (300,300),
        batch_size = 128,
        class_mode = 'binary'
    )
    model.fit(
        train_generator,
        epochs = 1
    )
    _, acc = model.evaluate(train_generator)
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


