"""
longest common subsequence
modified from the code snippets at
http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Python
cython -a lcs.pyx to output HTML
"""
import numpy as np

cimport cython
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef inline int int_max(int a, int b): return a if a >= b else b


@cython.boundscheck(False)
def longest_common_subsequence(X, Y):
    """Compute and return the longest common subsequence length
    X, Y are list of strings"""
    cdef int m = len(X) 
    cdef int n = len(Y)

    # use numpy array for memory efficiency with long sequences
    # lcs is bounded above by the minimum length of x, y
    assert min(m+1, n+1) < 65535

    #cdef np.ndarray[np.int32_t, ndim=2] C = np.zeros([m+1, n+1], dtype=np.int32)
    cdef np.ndarray[np.uint16_t, ndim=2] C = np.zeros([m+1, n+1], dtype=np.uint16)

    # convert X, Y to C++ standard containers
    cdef vector[string] xx = [elem.encode('utf-8') for elem in X]
    cdef vector[string] yy = [elem.encode('utf-8') for elem in Y]

    cdef int i, j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if xx[i-1] == yy[j-1]:
                C[i, j] = C[i-1, j-1] + 1
            else:
                C[i, j] = int_max(C[i, j-1], C[i-1, j])
    return C[-1, -1]
