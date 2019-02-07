"""
longest common subsequence
modified from the code snippets at
http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence#Python
cython -a lcs.pyx to output HTML
"""

cimport cython
from libcpp.vector cimport vector

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
    # cdef np.ndarray[np.uint16_t, ndim=2] C = np.zeros([m+1, n+1], dtype=np.uint16)
    cdef vector[uint] row = vector[uint](n+1, 0)
    cdef vector[vector[uint]] C = vector[vector[uint]](m+1, row)

    cdef int i, j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = int_max(C[i][j-1], C[i-1][j])
    return C[m][n]
