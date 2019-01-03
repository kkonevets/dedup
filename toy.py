import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from scipy.spatial.distance import cosine as dcosine

import tensorflow as tf

N = 1000
dim = 2
theta = np.pi/8
np.random.seed(0)


if __name__ == "__main__":

    vecs = np.random.rand(N, dim)
    dists = cosine_similarity(vecs, vecs)
    cond = dists > np.cos(theta)
    ixs = np.triu_indices(N, k=1)

    #################################################################

    # data = np.hstack((vecs[ixs[0]], vecs[ixs[1]]))
    # data = np.subtract(vecs[ixs[0]],  vecs[ixs[1]])
    data = np.array([[dcosine(a, b)]
                     for a, b in zip(vecs[ixs[0]], vecs[ixs[1]])])

    #################################################################

    target = cond[ixs]
    print(np.unique(target, return_counts=True))

    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = target.reshape(len(target), 1)
    data_y = onehot_encoder.fit_transform(integer_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        data, data_y, test_size=0.33, random_state=42, stratify=target)

    un = np.unique(y_test[:, 1], return_counts=True)
    counts = un[1]
    print('baseline: %s\n' % (counts[0]/(counts[0] + counts[1])))

    norm = Normalizer()
    X_train = norm.fit_transform(X_train)
    X_test = norm.transform(X_test)

    def gen_batches(X, y, batch_size=100):
        for i in range(0, X.shape[0], batch_size):
            yield X[i:i+batch_size], y[i:i+batch_size]

    #################################################################

    # Parameters
    learning_rate = 0.01
    batch_size = 128
    num_steps = int(X_train.shape[0]/batch_size)
    display_step = 100

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    num_input = data.shape[1]  # MNIST data input (img shape: 28*28)
    num_classes = 2  # MNIST total classes (0-9 digits)

    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])

    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.relu)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.layers.dense(layer_2, num_classes)
        return out_layer

    # Construct model
    logits = neural_net(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        train_batcher = gen_batches(X_train, y_train, batch_size)

        for step in range(1, num_steps+1):
            batch_x, batch_y = next(train_batcher)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " +
                      "{:.4f}".format(loss) + ", Training Accuracy= " +
                      "{:.3f}".format(acc))

        print("Optimization Finished!")

        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={X: X_test,
                                            Y: y_test}))
