import tools
import pandas as pd
import numpy as np
import tensorflow as tf
import tools
from tqdm import tqdm
import scipy

samples = tools.load_samples('../data/dedup/samples.npz')
tfidf = tools.do_unpickle('../data/dedup/tfidf_model.pkl')
corpus = tools.load_samples('../data/dedup/corpus.npz')

qids = samples[samples['synid'] == -1]['qid'].unique()
sids = samples[samples['synid'] != -1]['synid'].unique()
fids = samples['fid'].unique()


def get_id2text(tag, ids):
    id2text = {}
    for _id, text in corpus[corpus[tag].isin(ids)][[tag, 'text']].values:
        id2text[_id] = text
    return id2text


qid2text = get_id2text('qid', qids)
sid2text = get_id2text('synid', sids)
fid2text = get_id2text('fid', fids)


def input_fn(train=True, seed=0):
    cur_samples = samples[samples['train'] == int(train)]
    cur_samples = cur_samples.sample(frac=1, random_state=seed)

    qtexts, ftexts = [], []
    for row in cur_samples.itertuples():
        ftexts.append(fid2text[row.fid])

        if row.synid != -1:
            qtext = sid2text[row.synid]
        else:
            qtext = qid2text[row.qid]

        qtexts.append(qtext)

    qvecs = tfidf.transform(qtexts)
    fvecs = tfidf.transform(ftexts)
    data = scipy.sparse.hstack([qvecs, fvecs])
