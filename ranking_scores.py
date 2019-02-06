import numpy as np
from tools import ndcg_at_k


def test_tensor():
    import tensorflow as tf
    import tensorflow_ranking as tfr

    l = np.array([[1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0]])

    p = np.array([[1, 0, 0, 0, 0, 0]]*len(l))

    labels = tf.placeholder(tf.float32, shape=l.shape)
    predictions = tf.placeholder(tf.float32, shape=p.shape)
    score = tfr.metrics.normalized_discounted_cumulative_gain(
        labels, predictions, topn=2)

    initg = tf.initialize_all_variables()
    initl = tf.local_variables_initializer()

    with tf.Session() as sess:
        sess.run(initg)
        sess.run(initl)
        print(sess.run(score, feed_dict={predictions: p, labels: l}))


if __name__ == "__main__":
    l = [[1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 0]]

    scores = []
    for r in l:
        si = []
        for k in range(1, len(r) + 1):
            si.append(ndcg_at_k(r, k))
        scores.append(si)

    scores = np.array(scores)
    print(np.mean(scores, axis=0))

    for i in range(6):
        r = [0]*6
        r[i] = 1
        print(r, ndcg_at_k(r, 2))

    r = [0, 1, 0, 0, 0, 0]
    ndcg_at_k(r, 2)  # 0.63

    temp = [1, 1, 1, 0.63, 0.63, 0.63, 0.63,
            0.63, 0.63, 0.63, 0.63, 0.63]  # 0.725
    temp = [1, 1, 1, 0, 1, 1, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63]  # 0.73
    ndcg2 = np.mean(temp)
    print('ndcg2 %f' % ndcg2)

    l = [[1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 0]]

    scores = []
    for r in l:
        si = []
        for k in range(1, len(r) + 1):
            si.append(ndcg_at_k(r, k))
        scores.append(si)

    scores = np.array(scores)
    print(np.mean(scores, axis=0))
