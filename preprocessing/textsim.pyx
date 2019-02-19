cimport cython
from libcpp.vector cimport vector

import numpy as np
from fuzzywuzzy import fuzz
import py_stringmatching as sm
import Levenshtein
import textdistance as td

lcsstr = td.LCSStr()

aff = sm.Affine()
ed = sm.Editex()
oc = sm.OverlapCoefficient()
tvi = sm.TverskyIndex()
jac = sm.Jaccard()
qgram = QGram(2)
bd = sm.BagDistance()
gj = sm.GeneralizedJaccard()
gjw = sm.GeneralizedJaccard(
    sim_func=sm.JaroWinkler().get_raw_score, threshold=0.8)
me = sm.MongeElkan()
menw = sm.MongeElkan(sim_func=sm.NeedlemanWunsch().get_raw_score)


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


def lvn_dst(q, d):
    l = max(len(q), len(d))
    if l == 0:
        return 0
    return 1 - Levenshtein.distance(q, d)/l


def lcs_dst(q, d):
    m = max(max(len(q), len(d)), 1)
    return longest_common_subsequence(q, d)/m


def ratcliff_dst(q, d):
    return 1-td.ratcliff_obershelp.normalized_distance(q, d)


def tversky_dst(q, d):
    return 1-td.tversky.normalized_distance(q, d)


def matrix_ftrs(fn, q_split, d_split, tag):
    """
    fn - distance function
    """
    mat = np.array([[fn(qi, di) for di in d_split] for qi in q_split])
    arr = np.max(mat, axis=1)
    ftrs = [np.mean(arr), np.std(arr)]
    ftrs += [np.quantile(arr, v) for v in [0.25, 0.5, 0.75]]
    ftrs = {'%s_%d' % (tag, i+1): v for i, v in enumerate(ftrs)}
    return ftrs


func_map = {
    # 'bag_mat': lambda x, y: matrix_ftrs(bd.get_sim_score, x, y, 'bag'),
    'lvn_mat': lambda x, y: matrix_ftrs(lvn_dst, x, y, 'lvn'),
    'jaro_mat': lambda x, y: matrix_ftrs(Levenshtein.jaro, x, y, 'jaro'),
    'jaro_win_mat': lambda x, y: matrix_ftrs(Levenshtein.jaro_winkler, x, y, 'jaro_win'),
    'lcsseq_mat': lambda x, y: matrix_ftrs(lcs_dst, x, y, 'lcsseq'),
    # 'ratcliff_mat': lambda x, y: matrix_ftrs(ratcliff_dst, x, y, 'ratcliff'),
    'genjack_tok': gj.get_sim_score,
    'me_tok': me.get_raw_score,
    'menw_tok': menw.get_raw_score,
    'genjack_jw_tok': gjw.get_sim_score,
    'seqratio_tok': Levenshtein.seqratio,
    'setratio_tok': Levenshtein.setratio,
    'jaccard_tok': jac.get_sim_score,
    'lcsseq': lcs_dst,
    'lcsstr': lcsstr.normalized_similarity,
    'fuzz.ratio': fuzz.ratio,
    'fuzz.partial_ratio': fuzz.partial_ratio,
    'fuzz.token_sort_ratio': fuzz.token_sort_ratio,
    'fuzz.token_set_ratio': fuzz.token_set_ratio,
    'qgram': qgram.distance,
    'tversky_tok': tvi.get_sim_score,
    'overlap_tok': oc.get_sim_score,
    # 'editex': ed.get_sim_score,
    'prefix': td.prefix.normalized_distance,
    'postfix': td.postfix.normalized_distance,
    # 'affine': aff.get_raw_score
}


def get_sim_features(q_split, d_split):
    q = ' '.join(q_split)
    d = ' '.join(d_split)

    frac = len(q_split)/len(d_split) if len(d_split) else 0
    cfrac = len(q)/len(d) if len(d) else 0
    ftrs = {'q_len': len(q_split), 'd_len': len(d_split), 'q/d': frac,
            'q_clen': len(q), 'd_clen': len(d), 'c_q/d': cfrac}

    for k, func in func_map.items():
        if '_mat' in k:
            ftrs.update(func(q_split, d_split))
        elif '_tok' in k:
            ftrs[k] = func(q_split, d_split)
        else:
            ftrs[k] = func(q, d)

    return ftrs


def get_extra_ftrs(q_split, d_split):
    ftrs = {}
    # 'needleman_wunsch': td.needleman_wunsch.normalized_distance
    # 'gotoh': td.gotoh.normalized_distance
    return ftrs


