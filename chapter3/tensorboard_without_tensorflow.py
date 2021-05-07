from absl import app
import tensorflow as tf
import pandas as pd


!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
df = pd.read_csv('winequality-red.csv', delimiter=';')


with tf.summary.create_file_writer("logs/text").as_default():
    for col in df.columns:
        step = 0
        for val in df[col]:
            s = tf.summary.text(name=col, data=str(val), step=step)
            step = step + 1


