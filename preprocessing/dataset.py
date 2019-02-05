import tools
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import io
from sklearn.model_selection import train_test_split
from preprocessing import textsim
from tqdm import tqdm
import multiprocessing as mp
from itertools import islice
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine
from gensim.models import FastText


INFO_COLUMNS = ['qid', 'synid', 'fid', 'target']
COLNAMES = INFO_COLUMNS + ['score', 'ix']


def get_id2text(corpus, tag, ids):
    id2text = {}
    for _id, text in corpus[corpus[tag].isin(ids)][[tag, 'text']].values:
        id2text[_id] = text
    return id2text


def input_data(cur_samples, fid2text, sid2text, qid2text):
    q_terms, d_terms = [], []
    rows = []
    for row in cur_samples.itertuples():
        f_splited = fid2text[row.fid].split()
        if row.synid != -1:
            qtext = sid2text[row.synid]
        else:
            qtext = qid2text[row.qid]

        q_splited = qtext.split()
        if row.target == 0 and ' '.join(f_splited) == ' '.join(q_splited):
            continue

        if len(q_splited) * len(f_splited) == 0:
            continue

        rows.append(row.Index)
        d_terms.append(f_splited)
        q_terms.append(q_splited)

    # TODO: add DNN features: brands ...

    cur_samples = cur_samples.loc[rows]
    values = cur_samples[COLNAMES]
    assert len(values) == len(q_terms)
    return q_terms,  d_terms, values.values


def sim_worker(extra, tup):
    q_terms, d_terms, info = tup
    if extra:
        ftrs = textsim.get_extra_ftrs(q_terms, d_terms)
    else:
        ftrs = textsim.get_sim_features(q_terms, d_terms)
    values = list(info) + list(ftrs.values())
    columns = COLNAMES + list(ftrs.keys())
    return values, columns


def get_similarity_features(data, output_file, extra=False):
    columns = None

    def feeder(data):
        for tup in zip(*data):
            yield tup

    wraper = partial(sim_worker, extra)

    vals = []
    with mp.Pool(mp.cpu_count(), maxtasksperchild=100000) as p:
        with tqdm(total=len(data[0])) as pbar:
            for values, columns in tqdm(p.imap_unordered(wraper, feeder(data))):
                vals.append(values)
                pbar.update()

    vals = np.array(vals, dtype=np.float32)
    np.savez(output_file, vals=vals, columns=columns)
    return vals, columns


def compute_tfidf_dists(train_data, test_data):
    tfidf_model = tools.do_unpickle('../data/dedup/tfidf_model.pkl')

    def get_dists(data, fname):
        qs = [' '.join(q_terms) for q_terms in data[0]]
        ds = [' '.join(d_terms) for d_terms in data[1]]
        qvecs = tfidf_model.transform(qs)
        dvecs = tfidf_model.transform(ds)

        dists = paired_cosine_distances(qvecs, dvecs)
        ixs = data[2][:, :3]
        np.savez(fname, dists=np.hstack([ixs, np.array([dists]).T]))

    get_dists(train_data, '../data/dedup/train_tfidf_cosine.npz')
    get_dists(test_data, '../data/dedup/test_tfidf_cosine.npz')


def compute_fasttext_dists():
    train_data_raw = tools.do_unpickle('../data/dedup/train_data_raw.pkl')
    test_data_raw = tools.do_unpickle('../data/dedup/test_data_raw.pkl')

    model = FastText.load_fasttext_format('../data/dedup/cc.ru.300.bin')
    # model.wv.most_similar('ватт')

    def get_dists(data, fname):
        dists = []
        for q_terms, d_terms, _ixs in tqdm(zip(data[0], data[1], data[2]), total=len(data[0])):
            qvecs = [model.wv[term] for term in q_terms if
                     term.isalpha() and len(term) > 2 and term in model.wv]
            dvecs = [model.wv[term] for term in d_terms if
                     term.isalpha() and len(term) > 2 and term in model.wv]
            qmean = np.mean(qvecs, axis=0)
            dmean = np.mean(dvecs, axis=0)
            if len(qvecs) and len(dvecs):
                paired = pairwise_distances(qvecs, dvecs, metric='cosine')
                mins = np.min(paired, axis=1)
                mean, median, std = np.mean(mins), \
                    np.median(mins), np.std(mins)
            else:
                mean, median, std = -1, -1, -1
            dists.append(list(_ixs[:3]) + [cosine(qmean, dmean),
                                           mean, median, std])
        np.savez(fname, dists=dists)

    get_dists(train_data_raw, '../data/dedup/train_fasttext_cosine.npz')
    get_dists(test_data_raw, '../data/dedup/test_fasttext_cosine.npz')


#########################################################################


def letor_producer(X, qst):
    _id = 0
    qid_prev, synid_prev = None, None
    for (qid, synid, target), row in tqdm(zip(qst, X), total=len(X)):
        if (qid_prev, synid_prev) != (qid, synid):
            _id += 1
        qid_prev, synid_prev = qid, synid
        yield target, _id, row


def to_letor(X, qst, fname):
    with open(fname, 'w') as f:
        for target, _id, row in letor_producer(X, qst):
            s = '%d qid:%d' % (target, _id)
            _sft = ' '.join(['%d:%f' % (i + 1, v)
                             for i, v in enumerate(row)])
            s = ' '.join([s, _sft, '\n'])
            f.write(s)


def letor_prepare(train_sim_ftrs, test_sim_ftrs):
    train_sim_ftrs.sort_values(['qid', 'synid'], inplace=True)
    test_sim_ftrs.sort_values(['qid', 'synid'], inplace=True)

    value_cols = [c for c in train_sim_ftrs.columns if c not in INFO_COLUMNS]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_sim_ftrs[value_cols])
    X_test = scaler.transform(test_sim_ftrs[value_cols])

    qst_train = train_sim_ftrs[['qid', 'synid', 'target']].values
    qst_test = test_sim_ftrs[['qid', 'synid', 'target']].values

    return X_train, qst_train, X_test, qst_test


def save_letor_txt(train_sim_ftrs, test_sim_ftrs, vali=False):
    X_train, qst_train, X_test, qst_test = letor_prepare(
        train_sim_ftrs, test_sim_ftrs)

    to_letor(X_test, qst_test, '../data/dedup/test_letor.txt')

    if vali:
        hashtag = qst_train[:, :2]  # ['qid', 'synid']
        hashtag = pd.Series(map(tuple, hashtag))
        hash_train, hash_vali = train_test_split(
            hashtag.unique(), test_size=0.1, random_state=42)
        cond = hashtag.isin(hash_train).values
        to_letor(X_train[cond], qst_train[cond],
                 '../data/dedup/train_letor.txt')
        to_letor(X_train[~cond], qst_train[~cond],
                 '../data/dedup/vali_letor.txt')
    else:
        to_letor(X_train, qst_train, '../data/dedup/train_letor.txt')


def _int32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def to_example(data, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    q_terms, d_terms, info = data
    labels = info[:, -1]
    for q, d, l in zip(q_terms, d_terms, labels):
            # Create a feature
        feature = {
            'q_terms': _bytes_feature([tf.compat.as_bytes(qi) for qi in q]),
            'd_terms': _bytes_feature([tf.compat.as_bytes(di) for di in d]),
            'labels': _int32_feature(int(l)),
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def to_letor_example(train_sim_ftrs, test_sim_ftrs):
    X_train, qst_train, X_test, qst_test = letor_prepare(
        train_sim_ftrs, test_sim_ftrs)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    def save_one(X, qst, filename):
        writer = tf.python_io.TFRecordWriter(filename)
        _id_prev = None
        for target, _id, row in letor_producer(X, qst):
            # Create a feature
            feature = {
                'qid': _int32_feature(int(_id)),
                'x': tf.train.Feature(float_list=tf.train.FloatList(value=row)),
                'labels': _int32_feature(int(target)),
            }
            # Create an example protocol buffer
            example = tf.train.Example(
                features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

        writer.close()

    save_one(X_train, qst_train, '../data/dedup/train_letor.tfrecord')
    save_one(X_test, qst_test, '../data/dedup/test_letor.tfrecord')


def dists_from_numpy(fname, tag):
    df = np.load(fname)['dists']
    df = pd.DataFrame(df)
    columns = ['qid', 'synid', 'fid']
    columns += ['%s%d' % (tag, i) for i in range(df.shape[1]-3)]
    df.columns = columns
    df.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
    return df


def load_sim_ftrs(with_extra=True):
    test_sim_ftrs = tools.load_samples(
        '../data/dedup/test_sim_ftrs.npz', key='vals')
    train_sim_ftrs = tools.load_samples(
        '../data/dedup/train_sim_ftrs.npz', key='vals')

    test_sim_ftrs.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
    train_sim_ftrs.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)

    test_tfidf_cos = dists_from_numpy(
        '../data/dedup/test_tfidf_cosine.npz', 'tfidf')
    train_tfidf_cos = dists_from_numpy(
        '../data/dedup/train_tfidf_cosine.npz', 'tfidf')

    test_sim_ftrs = test_sim_ftrs.merge(
        test_tfidf_cos, on=['qid', 'synid', 'fid'])
    train_sim_ftrs = train_sim_ftrs.merge(
        train_tfidf_cos, on=['qid', 'synid', 'fid'])

    test_ft_cos = dists_from_numpy(
        '../data/dedup/test_fasttext_cosine.npz', 'ft')
    train_ft_cos = dists_from_numpy(
        '../data/dedup/train_fasttext_cosine.npz', 'ft')

    test_sim_ftrs = test_sim_ftrs.merge(
        test_ft_cos, on=['qid', 'synid', 'fid'])
    train_sim_ftrs = train_sim_ftrs.merge(
        train_ft_cos, on=['qid', 'synid', 'fid'])

    test_sim_ftrs.fillna(-1, inplace=True)
    train_sim_ftrs.fillna(-1, inplace=True)

    if with_extra:
        print("WARNING: Loading extra features!")
        test_extra = tools.load_samples(
            '../data/dedup/test_sim_ftrs_extra.npz', key='vals')
        usecols = ['qid', 'synid', 'fid'] + \
            list(set(test_extra.columns).difference(COLNAMES))
        test_extra = test_extra[usecols]
        train_extra = tools.load_samples(
            '../data/dedup/train_sim_ftrs_extra.npz', key='vals')
        train_extra = train_extra[usecols]

        # unite with extra
        test_sim_ftrs = test_sim_ftrs.merge(
            test_extra, on=['qid', 'synid', 'fid'])
        train_sim_ftrs = train_sim_ftrs.merge(
            train_extra, on=['qid', 'synid', 'fid'])

    return train_sim_ftrs, test_sim_ftrs


def main():
    samples = tools.load_samples('../data/dedup/samples.npz')

    # exclude samples not found in TOP
    synids_exclude = set(samples[samples['ix'] == -1]['synid'].unique())
    synids_exclude.discard(-1)
    samples = samples[~samples['synid'].isin(synids_exclude)]
    qids_exclude = samples[samples['ix'] == -1]['qid'].unique()
    samples = samples[~samples['qid'].isin(qids_exclude)]

    qids = samples[samples['synid'] == -1]['qid'].unique()
    sids = samples[samples['synid'] != -1]['synid'].unique()
    fids = samples['fid'].unique()

    raw = True
    corpus = tools.load_samples(
        '../data/dedup/corpus%s.npz' % ('_raw' if raw else ''))

    qid2text = get_id2text(corpus, 'qid', qids)
    sid2text = get_id2text(corpus, 'synid', sids)
    fid2text = get_id2text(corpus, 'fid', fids)

    vals = corpus[corpus['train'] != 0]['text'].values
    informative_terms = set([w for s in vals for w in s.split()])
    with io.open('../data/dedup/vocab.txt', 'w', encoding='utf8') as f:
        for term in informative_terms:
            f.write(term + '\n')

    train_data = input_data(
        samples[samples['train'] == 1], fid2text, sid2text, qid2text)
    test_data = input_data(
        samples[samples['train'] == 0], fid2text, sid2text, qid2text)
    tools.do_pickle(train_data, '../data/dedup/train_data%s.pkl' %
                    ('_raw' if raw else ''))
    tools.do_pickle(test_data, '../data/dedup/test_data%s.pkl' %
                    ('_raw' if raw else ''))

    #########################################################################

    train_data = tools.do_unpickle('../data/dedup/train_data.pkl')
    test_data = tools.do_unpickle('../data/dedup/test_data.pkl')
    compute_tfidf_dists(train_data, test_data)

    compute_fasttext_dists()

    # to_example(train_data, '../data/dedup/train.tfrecord')
    # to_example(test_data, '../data/dedup/test.tfrecord')

    # sub_test = [v[:1000] for v in test_data]
    # vals, columns = test_sim_ftrs
    test_sim_ftrs = get_similarity_features(
        test_data, '../data/dedup/test_sim_ftrs.npz')
    train_sim_ftrs = get_similarity_features(
        train_data, '../data/dedup/train_sim_ftrs.npz')
    # test_extra = get_similarity_features(
    #     test_data, '../data/dedup/test_sim_ftrs_extra.npz', True)
    # train_extra = get_similarity_features(
    #     train_data, '../data/dedup/train_sim_ftrs_extra.npz', True)

    train_sim_ftrs, test_sim_ftrs = load_sim_ftrs(with_extra=False)
    save_letor_txt(train_sim_ftrs, test_sim_ftrs, vali=True)

    # to_letor_example(train_sim_ftrs, test_sim_ftrs)

    # output_file = '../data/dedup/test_sim_ftrs.npz'
    # vals = test_sim_ftrs.values
    # vals = np.array(vals, dtype=np.float32)
    # np.savez(output_file, vals=vals, columns=test_sim_ftrs.columns)


if __name__ == "__main__":
    pass
