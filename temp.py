import numpy as np


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


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
            si.append(ndcg_at_k(r, k, method=1))
        scores.append(si)

    scores = np.array(scores)
    print(np.mean(scores, axis=0))

    for i in range(6):
        r = [0]*6
        r[i] = 1
        print(r, ndcg_at_k(r, 2, method=1))

    r = [0, 1, 0, 0, 0, 0]
    ndcg_at_k(r, 2, method=1)  # 0.63

    temp = [1, 1, 1, 0.63, 0.63, 0.63, 0.63,
            0.63, 0.63, 0.63, 0.63, 0.63]  # 0.725
    temp = [1, 1, 1, 0, 1, 1, 0.63, 0.63, 0.63, 0.63, 0.63, 0.63]  # 0.73
    ndcg2 = np.mean(temp)
    print('ndcg2 %f' % ndcg2)
