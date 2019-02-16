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
from dedup.textsim import get_sim_features, func_map
import os
import multiprocessing as mp
import h5py


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


def append_h5(fname, vals, columns):
    vals = np.array(vals, dtype=np.float32)
    with h5py.File(fname, 'a') as hf:
        if len(hf.keys()) == 0:
            hf.create_dataset('ftrs', data=vals,
                              maxshape=(None, vals.shape[1]),
                              dtype='f', chunks=True)
        else:
            hf['ftrs'].resize(hf['ftrs'].shape[0] + vals.shape[0], axis=0)

        hf['ftrs'][-vals.shape[0]:] = vals
        hf['ftrs'].attrs['columns'] = columns
        hf.flush()


def extract_similarity_features(data_gen, colnames, output_file):
    if os.path.isfile(output_file):
        os.remove(output_file)

    vals = []
    max_len = 200000
    maxtasksperchild = int(max_len/mp.cpu_count())
    with mp.Pool(mp.cpu_count(), maxtasksperchild=maxtasksperchild) as p:
        for values, columns in p.imap_unordered(sim_worker, data_gen):
            vals.append(values)
            if len(vals) >= max_len:
                append_h5(output_file, vals, colnames + columns)
                vals = []

    # finaly
    if len(vals):
        append_h5(output_file, vals, colnames + columns)

        # for values, columns in map(wraper, data_gen):
        #     vals.append(values)
