import tools
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import io
from preprocessing import textsim
from tqdm import tqdm
import multiprocessing as mp
from itertools import islice
from sklearn.preprocessing import StandardScaler


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


def sim_worker(tup):
    q_terms, d_terms, info = tup
    q = ' '.join(q_terms)
    d = ' '.join(d_terms)
    ftrs = textsim.get_sim_features(q, d)
    values = list(info) + list(ftrs.values())
    columns = COLNAMES + list(ftrs.keys())
    return values, columns


def get_similarity_features(data, output_file):
    columns = None

    def feeder(data):
        for tup in zip(*data):
            yield tup

    vals = []
    with mp.Pool(mp.cpu_count(), maxtasksperchild=5000) as p:
        with tqdm(total=len(data[0])) as pbar:
            for values, columns in tqdm(p.imap_unordered(sim_worker, feeder(data))):
                vals.append(values)
                pbar.update()

    vals = np.array(vals, dtype=np.float32)
    np.savez(output_file, vals=vals, columns=columns)
    return vals, columns

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


def save_letor_txt(train_sim_ftrs, test_sim_ftrs):
    X_train, qst_train, X_test, qst_test = letor_prepare(
        train_sim_ftrs, test_sim_ftrs)

    to_letor(X_train, qst_train, '../data/dedup/train_letor.txt')
    to_letor(X_test, qst_test, '../data/dedup/test_letor.txt')


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

    X_train = X_train.astype(np.float)
    X_test = X_test.astype(np.float)

    def save_one(X, qst, filename):
        writer = tf.python_io.TFRecordWriter(filename)
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


def main():
    samples = tools.load_samples('../data/dedup/samples.npz')

    # exclude samples not found in TOP
    synids_exclude = set(samples[samples['ix'] == -1]['synid'].unique())
    synids_exclude.remove(-1)
    samples = samples[~samples['synid'].isin(synids_exclude)]
    qids_exclude = samples[samples['ix'] == -1]['qid'].unique()
    samples = samples[~samples['qid'].isin(qids_exclude)]

    qids = samples[samples['synid'] == -1]['qid'].unique()
    sids = samples[samples['synid'] != -1]['synid'].unique()
    fids = samples['fid'].unique()

    corpus = tools.load_samples('../data/dedup/corpus.npz')

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
    tools.do_pickle(train_data, '../data/dedup/train_data.pkl')
    tools.do_pickle(test_data, '../data/dedup/test_data.pkl')

    #########################################################################

    # train_data = tools.do_unpickle('../data/dedup/train_data.pkl')
    # test_data = tools.do_unpickle('../data/dedup/test_data.pkl')

    to_example(train_data, '../data/dedup/train.tfrecord')
    to_example(test_data, '../data/dedup/test.tfrecord')

    # sub_test = [v[:1000] for v in test_data]
    # vals, columns = test_sim_ftrs
    test_sim_ftrs = get_similarity_features(
        test_data, '../data/dedup/test_sim_ftrs.npz')
    train_sim_ftrs = get_similarity_features(
        train_data, '../data/dedup/train_sim_ftrs.npz')

    # train_sim_ftrs = tools.load_samples(
    #     '../data/dedup/train_sim_ftrs.npz', key='vals')
    # test_sim_ftrs = tools.load_samples(
    #     '../data/dedup/test_sim_ftrs.npz', key='vals')

    save_letor_txt(train_sim_ftrs, test_sim_ftrs)

    to_letor_example(train_sim_ftrs, test_sim_ftrs)


if __name__ == "__main__":
    pass
