import tensorflow as tf

#########################################################################

batch_size = 128
embedding_dim = 100

train_path = '../data/dedup/train.tfrecord'
test_path = '../data/dedup/test.tfrecord'
vocab_file = '../data/dedup/vocab.txt'

#########################################################################


def _parse_function(record):
    """Extracts features and labels.

    Args:
      record: File path to a TFRecord file    
    Returns:
      A `tuple` `(labels, features)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features = {
        # terms are strings of varying lengths
        "q_terms": tf.VarLenFeature(dtype=tf.string),
        "d_terms": tf.VarLenFeature(dtype=tf.string),
        # labels are 0 or 1
        "labels": tf.FixedLenFeature(shape=[1], dtype=tf.int64)
    }

    parsed_features = tf.parse_single_example(record, features)

    q_terms = parsed_features['q_terms'].values
    d_terms = parsed_features['d_terms'].values
    labels = parsed_features['labels']

    return {'q_terms': q_terms, 'd_terms': d_terms}, labels


# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def _input_fn(input_filenames, batch_size=128, num_epochs=1, shuffle=True, seed=0):

    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(_parse_function)

    if shuffle:
        ds = ds.shuffle(10000, seed=seed)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(batch_size, ds.output_shapes)

    ds = ds.repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

#########################################################################


q_terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
    "q_terms", vocab_file)
d_terms_feature_column = tf.feature_column.categorical_column_with_vocabulary_file(
    "d_terms", vocab_file)
wide_columns = [q_terms_feature_column,  d_terms_feature_column]

q_terms_embedding_column = tf.feature_column.embedding_column(
    q_terms_feature_column, dimension=embedding_dim)
d_terms_embedding_column = tf.feature_column.embedding_column(
    d_terms_feature_column, dimension=embedding_dim)

deep_columns = [q_terms_embedding_column, d_terms_embedding_column]

my_optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

classifier = tf.estimator.DNNClassifier(
    feature_columns=deep_columns,
    hidden_units=[200, 200],
    optimizer=my_optimizer,
    model_dir="./model/deep"
)


# classifier = tf.estimator.LinearClassifier(
#     feature_columns=wide_columns,
#     # optimizer=tf.train.FtrlOptimizer(
#     #     learning_rate=0.9,
#     #     # l1_regularization_strength=0.001
#     # ),
#     # model_dir="./model/wide"
# )

#########################################################################

classifier.train(
    input_fn=lambda: _input_fn(
        train_path, batch_size=batch_size, num_epochs=5),
)

#########################################################################

evaluation_metrics = classifier.evaluate(
    input_fn=lambda: _input_fn(
        train_path, batch_size=batch_size, num_epochs=1))
print("\nTraining set metrics:")
for m in evaluation_metrics:
    print(m, evaluation_metrics[m])
print("---")

# evaluation_metrics = classifier.evaluate(
#     input_fn=lambda: _input_fn(test_path, batch_size=batch_size, num_epochs=1))

# print("\nTest set metrics:")
# for m in evaluation_metrics:
#     print(m, evaluation_metrics[m])
# print("---")
