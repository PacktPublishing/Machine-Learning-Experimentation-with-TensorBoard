!pip install fairness_indicators
!pip install tensorboard-plugin-fairness-indicators
!pip install tensorflow-serving-api==2.4.1

import pandas as pd
import tensorflow as tf
import numpy as np

import tempfile
from tensorboard_plugin_fairness_indicators import summary_v2

import tensorflow_model_analysis as tfma
from google.protobuf import text_format

!wget https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
df = pd.read_csv('winequality-red.csv', delimiter=';')
df['alcohol_range'] = list(map(lambda x: 'low' if x < 10.0 else 'high', df['alcohol']))
df['pred'] = np.random.randint(low=3, high=9, size=df.shape[0])
df['pred'] = df['pred'] / np.max(df['quality'])
df['quality'] = df['quality'] / np.max(df['quality'])


data_root = tempfile.mkdtemp(prefix='wine-data')
eval_config = text_format.Parse("""
  model_specs {
    prediction_key: 'pred',
    label_key: 'quality'
  }
  metrics_specs {
    metrics {class_name: "AUC"}
    metrics {
      class_name: "FairnessIndicators"
      config: '{"thresholds": [0.3, 0.9]}'
    }
  }
  slicing_specs {
    feature_keys: 'alcohol_range'
  }
  slicing_specs {}
  """, tfma.EvalConfig())


# Run TensorFlow Model Analysis.
eval_result = tfma.analyze_raw_data(
  data=df,
  eval_config=eval_config,
  output_path=data_root)

writer = tf2.summary.create_file_writer('./logs/fairness')
with writer.as_default():
  summary_v2.FairnessIndicators(data_root, step=1)
writer.close()



