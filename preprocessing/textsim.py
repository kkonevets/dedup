import pandas as pd
import tools
import time
from tqdm import tqdm

import numpy as np
from fuzzywuzzy import fuzz
import py_stringmatching as sm
from pprint import pprint
import Levenshtein
import textdistance as td
from similarity.ngram import NGram
from similarity.qgram import QGram
from dedup import lcs

lcsstr = td.LCSStr()

qgram = QGram(2)
twogram = NGram(2)
threegram = NGram(3)


def lvn_dst(q, d):
    l = max(len(q), len(d))
    if l == 0:
        return 0
    return 1 - Levenshtein.distance(q, d)/l


def lcs_dst(q, d):
    n = lcs.longest_common_subsequence(q, d)[-1, -1]
    return n/max(len(q), len(d))


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


def get_sim_features(q_split, d_split):
    q = ' '.join(q_split)
    d = ' '.join(d_split)

    frac = len(q_split)/len(d_split) if len(d_split) else 0
    cfrac = len(q)/len(d) if len(d) else 0
    ftrs = {'q_len': len(q_split), 'd_len': len(d_split), 'q/d': frac,
            'q_clen': len(q), 'd_clen': len(d), 'c_q/d': cfrac}

    ftrs.update(matrix_ftrs(lvn_dst, q_split, d_split, 'lvn'))
    ftrs.update(matrix_ftrs(Levenshtein.jaro, q_split, d_split, 'jaro'))
    ftrs.update(matrix_ftrs(Levenshtein.jaro_winkler,
                            q_split, d_split, 'jaro_win'))
    ftrs.update(matrix_ftrs(lambda x, y: lcs_dst(x, y),
                            q_split, d_split, 'lcsseq'))
    ftrs.update(matrix_ftrs(lambda x, y: 1-td.ratcliff_obershelp.normalized_distance(x, y),
                            q_split, d_split, 'ratcliff'))
    ftrs.update(matrix_ftrs(lambda x, y: 1-td.tversky.normalized_distance(x, y),
                            q_split, d_split, 'tversky'))

    ftrs['seqratio'] = Levenshtein.seqratio(q_split, d_split)
    ftrs['setratio'] = Levenshtein.setratio(q_split, d_split)
    ftrs['jaccard'] = td.jaccard.normalized_distance(q, d)
    ftrs['lcsseq'] = lcs_dst(q, d)
    ftrs['lcsstr'] = lcsstr.normalized_similarity(q, d)
    ftrs['twogram'] = twogram.distance(q, d)
    ftrs['threegram'] = threegram.distance(q, d)

    ftrs['fuzz.ratio'] = fuzz.ratio(q, d)/100.
    ftrs['fuzz.partial_ratio'] = fuzz.partial_ratio(q, d)/100.
    ftrs['fuzz.token_sort_ratio'] = fuzz.token_sort_ratio(q, d)/100.
    ftrs['fuzz.token_set_ratio'] = fuzz.token_set_ratio(q, d)/100.

    ftrs['qgram'] = qgram.distance(q, d)
    ftrs['tversky'] = td.tversky.normalized_distance(q_split, d_split)
    ftrs['overlap'] = td.overlap.normalized_distance(q_split, d_split)

    ftrs['prefix'] = td.prefix.normalized_distance(q, d)
    ftrs['postfix'] = td.postfix.normalized_distance(q, d)

    return ftrs


def get_extra_ftrs(q_split, d_split):
    ftrs = {}
    # 'needleman_wunsch': td.needleman_wunsch.normalized_distance
    # 'gotoh': td.gotoh.normalized_distance
    return ftrs


def compare(q1, d1, q2, d2):
    s1 = get_sim_features(q1, d1)
    s2 = get_sim_features(q2, d2)
    df = pd.DataFrame(list(s1.values()))
    df.index = s1.keys()
    df['s2'] = list(s2.values())
    df.columns = ['s1', 's2']
    df['diff'] = df['s2'] - df['s1']
    return df


def performance():
    def compute(fn, q, d):
        start = time.time()
        fn(q, d)
        end = time.time()
        return end - start

    corpus = tools.load_samples('../data/dedup/corpus.npz')

    fmap = {'lvn': lvn_dst}

    times = {}
    d = 'test'
    for q in tqdm(corpus['text'].values[:1000]):
        for k, fn in fmap.items():
            d_split = d.split()

            t = compute(fn, q, d)

            t_prev = times.get(k, 0)
            times[k] = t_prev + t
        d = q

    times = pd.DataFrame.from_dict(times, orient='index')
    times.columns = ('time',)
    times.sort_values('time', inplace=True, ascending=False)
    print(times)


def test():
    q = 'мнямс конс соб телятина ветчина 200.0 грамм мнямс'
    d = '200.0 грамм конс   тел ветчин мнямс мнямс'
    q_split, d_split = q.split(), d.split()
    s = get_sim_features(q_split, d_split)
    pprint(s)
