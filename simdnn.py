import tensorflow as tf
import numpy as np
import pandas as pd
import shutil
import os
from sklearn.preprocessing import Normalizer


#########################################################################

batch_size = 128
num_epochs = 20
epochs_between_evals = 4

epochs = (num_epochs//epochs_between_evals)*[epochs_between_evals]
mod = num_epochs % epochs_between_evals
if mod:
    epochs += [mod]


def load_data(fname):
    npzfile = np.load(fname)
    vals = pd.DataFrame(npzfile['vals'])
    vals.columns = npzfile['columns']
    return vals, npzfile['labels']


data_train, y_train = load_data('../data/dedup/train_sim_ftrs.npz')
data_test, y_test = load_data('../data/dedup/test_sim_ftrs.npz')

cols = [c for c in data_train.columns if c not in {
    'qid', 'synid', 'fid', 'score', 'ix', 'target'}]

norm = Normalizer()
X_train = norm.fit_transform(data_train[cols])
X_test = norm.transform(data_test[cols])

model_dir = "./model/simdnn"

#########################################################################


def _input_fn(train=True, num_epochs=1, shuffle=False, seed=0):
    if train:
        ds = tf.data.Dataset.from_tensor_slices(({'x': X_train}, y_train))
    else:
        ds = tf.data.Dataset.from_tensor_slices(({'x': X_test}, y_test))

    if shuffle:
        ds = ds.shuffle(10000, seed=seed)

    ds = ds.batch(batch_size).repeat(num_epochs)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


#########################################################################

# feature_columns = [tf.feature_column.numeric_column(key=k) for k in cols]
feature_columns = [tf.feature_column.numeric_column(
    "x", shape=X_train.shape[1])]

my_optimizer = tf.train.AdagradOptimizer(
    learning_rate=0.01,
    # l1_regularization_strength=0.001,
)

classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[100, 100],
    # dropout=0.2,
    optimizer=my_optimizer,
    model_dir=model_dir
)

#########################################################################


def do_eval(name):
    evaluation_metrics = classifier.evaluate(
        input_fn=lambda: _input_fn(
            name == 'train', num_epochs=1, shuffle=False),
        name=name
    )
    print("\n%s set metrics:" % name.upper())
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")


if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

for nep in epochs:
    tf.logging.set_verbosity(tf.logging.INFO)
    classifier.train(lambda: _input_fn(
        True, num_epochs=nep, shuffle=True))

    tf.logging.set_verbosity(tf.logging.ERROR)

    do_eval('train')
    do_eval('test')
