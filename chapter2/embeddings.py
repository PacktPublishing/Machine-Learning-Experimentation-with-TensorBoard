import os
from tensorboard.plugins import projector
import tensorflow as tf
import tensorflow_datasets as tfds


(training_dataset, val_dataset), information = tfds.load("imdb_reviews/subwords8k", split=(tfds.Split.TRAIN, tfds.Split.TEST), as_supervised=True, with_info=True)

encoder = information.features["text"].encoder


training_dataset.shuffle(1000).padded_batch(10, padded_shapes=((None,), ()))
val_batches = val_dataset.shuffle(1000).padded_batch(10, padded_shapes=((None,), ()))
next_train_batch, next_train_labels = next(iter(training_batches))


embedding_layer = tf.keras.layers.Embedding(encoder.vocab_size, 16)


model = tf.keras.Sequential([embedding_layer,
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1),
])

model.compile(
    optimizer="adam",
    metrics=["accuracy"],
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

model.fit(
    training_batches, epochs=2,
    validation_data=val_batches,
    validation_steps=15
)


log_dir='/logs/embeddings/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


f = open(os.path.join(log_dir, 'meta.tsv'), "w")
for sw in encoder.subwords:
    f.write(sw + '\n')

subwords_length = len(encoder.subwords)
for i in range(1, encoder.vocab_size - subwords_length):
    f.write('unknown #' + str(i) + '\n')

embedding_weights = tf.Variable(model.layers[0].get_weights()[0][1:])
tf_checkpoint = tf.train.Checkpoint(embedding=embedding_weights)
tf_checkpoint.save(os.path.join(log_dir, "embeddings_checkpoint.ckpt"))
 

projector_config = projector.ProjectorConfig()
projector_embedding = projector_config.embeddings.add()

projector_embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
projector_embedding.metadata_path = 'meta.tsv'
projector.visualize_embeddings(log_dir, projector_config)






