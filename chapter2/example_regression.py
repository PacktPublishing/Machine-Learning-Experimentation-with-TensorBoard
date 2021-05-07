import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import model_selection


!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
df = pd.read_csv('winequality-red.csv', delimiter=';')


train, val = model_selection.train_test_split(df, train_size=0.8)
train_y = train['quality']
train_x = train.drop(columns=['quality'])

val_y = val['quality']
val_x = val.drop(columns=['quality'])


logdir = "logs/scalars/"
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model = tf.keras.Sequential([
                     tf.keras.layers.Dense(64, input_dim=11, activation='relu'),
                     tf.keras.layers.Dense(32, activation='relu'),
                     tf.keras.layers.Dense(16, activation='relu'),
                     tf.keras.layers.Dense(1)
])

model.compile(metrics=['mse'],
              loss='mse',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

model.fit(
   train_x,
   train_y,
   batch_size=len(train_y),
   epochs=1000,
   validation_data=(val_x, val_y),
   callbacks=[tensorboard_callback]
)



