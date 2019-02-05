import pandas as pd
import tools
import time
from tqdm import tqdm

import numpy as np
from fuzzywuzzy import fuzz
import jellyfish
import py_stringmatching as sm
from pprint import pprint
import Levenshtein


def lvn_dst(q, d):
    l = max(len(q), len(d))
    if l == 0:
        return 0
    return 1 - Levenshtein.distance(q, d)/l


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
    frac = len(q_split)/len(d_split) if len(d_split) else 0
    ftrs = {'q_len': len(q_split), 'd_len': len(d_split), 'q/d': frac}

    ftrs.update(matrix_ftrs(lvn_dst, q_split, d_split, 'lvn'))
    ftrs.update(matrix_ftrs(Levenshtein.jaro, q_split, d_split, 'jaro'))
    ftrs.update(matrix_ftrs(Levenshtein.jaro_winkler,
                            q_split, d_split, 'jaro_win'))

    ftrs['seqratio'] = Levenshtein.seqratio(q_split, d_split)
    ftrs['setratio'] = Levenshtein.setratio(q_split, d_split)

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
    s = get_sim_features(q, d)
    df = pd.DataFrame.from_dict(s, orient='index')
    print(df)
