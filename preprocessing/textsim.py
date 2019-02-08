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
import multiprocessing as mp


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


def lvn_dst(q, d):
    l = max(len(q), len(d))
    if l == 0:
        return 0
    return 1 - Levenshtein.distance(q, d)/l


def lcs_dst(q, d):
    m = max(max(len(q), len(d)), 1)
    return lcs.longest_common_subsequence(q, d)/m


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

    times = {}
    d = 'test'
    for q in tqdm(corpus['text'].values[:1000]):
        q_split = q.split()
        d_split = d.split()
        for k, func in func_map.items():
            if '_mat' in k or '_tok' in k:
                t = compute(func, q_split, d_split)
            else:
                t = compute(func, q, d)

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


def sim_worker(tup):
    q_terms, d_terms, info = tup
    ftrs = get_sim_features(q_terms, d_terms)
    values = list(info) + list(ftrs.values())
    columns = list(ftrs.keys())
    return values, columns


def get_similarity_features(data_gen, colnames, output_file):
    columns = []
    vals = []
    with mp.Pool(mp.cpu_count(), maxtasksperchild=100000) as p:
        for values, columns in p.imap_unordered(sim_worker, data_gen):
            vals.append(values)

    # for values, columns in map(wraper, data_gen):
    #     vals.append(values)

    columns = colnames + columns

    vals = np.array(vals, dtype=np.float32)
    np.savez(output_file, vals=vals, columns=columns)
    return vals, columns
