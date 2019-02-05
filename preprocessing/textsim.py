import pandas as pd
import tools
import time
from tqdm import tqdm

from fuzzywuzzy import fuzz
import jellyfish
import textdistance as td
from similarity.optimal_string_alignment import OptimalStringAlignment
from similarity.damerau import Damerau
from similarity.metric_lcs import MetricLCS
from similarity.ngram import NGram
from similarity.qgram import QGram
import py_stringmatching as sm
from pprint import pprint

me = sm.MongeElkan()
menlw = sm.MongeElkan(sim_func=sm.NeedlemanWunsch().get_raw_score)
qgram = QGram(2)
twogram = NGram(2)
threegram = NGram(3)
fourgram = NGram(4)
damerau = Damerau()
metric_lcs = MetricLCS()
osa = OptimalStringAlignment()

m = {'fuzz.ratio': fuzz.ratio,
     'fuzz.partial_ratio': fuzz.partial_ratio,
     'fuzz.token_sort_ratio': fuzz.token_sort_ratio,
     'fuzz.token_set_ratio': fuzz.token_set_ratio,
     'jellyfish.jaro_winkler': jellyfish.jaro_winkler,
     'hamming': td.hamming.normalized_distance,
     'mlipns': td.mlipns.normalized_distance,
     'levenshtein': td.levenshtein.normalized_distance,
     'needleman_wunsch': td.needleman_wunsch.normalized_distance,
     'gotoh': td.gotoh.normalized_distance,
     'jaccard': td.jaccard.normalized_distance,
     'sorensen_dice': td.sorensen_dice.normalized_distance,
     'tversky': td.tversky.normalized_distance,
     'overlap': td.overlap.normalized_distance,
     'cosine': td.cosine.normalized_distance,
     'bag': td.bag.normalized_distance,
     'lcsstr': td.lcsstr.normalized_distance,
     'ratcliff': td.ratcliff_obershelp.normalized_distance,
     'bz2_ncd': td.bz2_ncd,
     'lzma_ncd': td.lzma_ncd.normalized_distance,
     'rle_ncd': td.rle_ncd.normalized_distance,
     'bwtrle_ncd': td.bwtrle_ncd.normalized_distance,
     'zlib_ncd': td.zlib_ncd.normalized_distance,
     'mra': td.mra.normalized_distance,
     'editex': td.editex.normalized_distance,
     'prefix': td.prefix.normalized_distance,
     'postfix': td.postfix.normalized_distance,
     'length': td.length.normalized_distance,
     'identity': td.identity.normalized_distance,
     'matrix': td.matrix.normalized_distance,
     'damerau': damerau.distance,
     'osa': osa.distance,
     'metric_lcs': metric_lcs.distance,
     'twogram': twogram.distance,
     'threegram': threegram.distance,
     'fourgram': fourgram.distance,
     'qgram': qgram.distance,
     }


def get_sim_features(q, d):
    q_split = q.split()
    d_split = d.split()

    frac = len(q)/len(d) if len(d) else 0
    ftrs = {'q_len': len(q), 'd_len': len(d), 'q/d': frac}

    ftrs.update({k: f(q, d) for k, f in m.items()})

    ftrs['me.get_raw_score'] = me.get_raw_score(q_split, d_split)
    ftrs['menlw'] = menlw.get_raw_score(q_split, d_split)

    return ftrs


def get_extra_ftrs(q, d):
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


def compute(fn, q, d):
    start = time.time()
    fn(q, d)
    end = time.time()
    return end - start


def performance():
    corpus = tools.load_samples('../data/dedup/corpus.npz')

    times = {}
    d = 'test'
    for q in tqdm(corpus['text'].values[:1000]):
        for k, fn in m.items():
            d_split = d.split()

            t = compute(fn, q, d)

            t_prev = times.get(k, 0)
            times[k] = t_prev + t

        q_split = q.split()
        t = compute(me.get_raw_score, q_split, d_split)
        t_prev = times.get(k, 0)
        times['me.get_raw_score'] = t_prev + t

        t = compute(menlw.get_raw_score, q_split, d_split)
        t_prev = times.get(k, 0)
        times['menlw.get_raw_score'] = t_prev + t

        d = q

    times = pd.DataFrame.from_dict(times, orient='index')
    times.columns = ('time',)
    times.sort_values('time', inplace=True, ascending=False)
    print(times)


def test():
    q = 'молоко'
    d = 'малока'
    s = get_sim_features(q, d)
    df = pd.DataFrame.from_dict(s, orient='index')
    print(df)
