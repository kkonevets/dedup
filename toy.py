import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.model_selection import train_test_split
import tensorflow as tf

N = 1000
dim = 5
theta = np.pi/8
np.random.seed(0)

vecs = np.random.rand(N, dim)
dists = cosine_similarity(vecs, vecs)
cond = dists > np.cos(theta)
ixs = np.triu_indices(N, k=1)
X = np.hstack((vecs[ixs[0]], vecs[ixs[1]]))
y = cond[ixs].astype(int)

print(np.unique(y, return_counts=True))


def gen_batches(X, batch_size=100):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

batcher = gen_batches(X, batch_size=100)
