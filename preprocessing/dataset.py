import tools
import numpy as np
import tensorflow as tf
import os
import io
from preprocessing import textsim
from tqdm import tqdm
import multiprocessing as mp
from itertools import islice


samples = tools.load_samples('../data/dedup/samples.npz')
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

vals = corpus[corpus['train'] != 0]['text'].values
informative_terms = set([w for s in vals for w in s.split()])
with io.open('../data/dedup/vocab.txt', 'w', encoding='utf8') as f:
    for term in informative_terms:
        f.write(term + '\n')

#########################################################################


def input_data(train=True):
    cur_samples = samples[samples['train'] == int(train)]

    fids, qids, q_terms, d_terms, scores, ixs, labels = [], [], [], [], [], [], []
    for row in cur_samples.itertuples():
        f_splited = fid2text[row.fid].split()
        if row.synid != -1:
            qtext = sid2text[row.synid]
        else:
            qtext = qid2text[row.qid]

        q_splited = qtext.split()
        if row.target == 0 and ' '.join(f_splited) == ' '.join(q_splited):
            continue

        if len(q_splited)*len(f_splited) == 0:
            continue

        fids.append(row.fid)
        qids.append("%d_%d" % (row.qid, row.synid))
        scores.append(row.score)
        ixs.append(row.ix)
        d_terms.append(f_splited)
        q_terms.append(q_splited)
        labels.append(row.target)

    # TODO: add DNN features: brands ...

    return qids, fids, np.array(q_terms),  np.array(d_terms), scores, ixs, labels


train_data = input_data(True)
test_data = input_data(False)
tools.do_pickle(train_data, '../data/dedup/train_data.pkl')
tools.do_pickle(test_data, '../data/dedup/test_data.pkl')


def worker(tup):
    qid, fid, q_terms, d_terms, score, ix, label = tup
    q = ' '.join(q_terms)
    d = ' '.join(d_terms)
    ftrs = textsim.get_sim_features(q, d)
    values = [qid, fid] + list(ftrs.values()) + [score, ix]
    columns = ['qid_synid', 'fid'] + list(ftrs.keys()) + ['score', 'ix']
    return values, columns


def get_similarity_features(data, output_file):
    labels = data[-1]
    columns = None

    def feeder(data):
        for tup in zip(*data):
            yield tup

    vals = []
    with mp.Pool(mp.cpu_count(), maxtasksperchild=10000) as p:
        with tqdm(total=len(data[0])) as pbar:
            for values, columns in tqdm(p.imap_unordered(worker, feeder(data))):
                vals.append(values)
                pbar.update()

    np.savez(output_file, vals=vals, labels=labels, columns=columns)
    return vals, labels, columns


# train_data = tools.do_unpickle('../data/dedup/train_data.pkl')
# test_data = tools.do_unpickle('../data/dedup/test_data.pkl')


# sub_test = [v[:1000] for v in test_data]
# vals, labels, columns = test_sim_ftrs
test_sim_ftrs = get_similarity_features(
    test_data, '../data/dedup/test_sim_ftrs.npz')
train_sim_ftrs = get_similarity_features(
    train_data, '../data/dedup/train_sim_ftrs.npz')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def to_example(data, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    qids, fids, q_terms, d_terms, scores, ixs, labels = data
    for q, d, l in zip(q_terms, d_terms, labels):
            # Create a feature
        feature = {
            'q_terms': _bytes_feature([tf.compat.as_bytes(qi) for qi in q]),
            'd_terms': _bytes_feature([tf.compat.as_bytes(di) for di in d]),
            'labels': _int64_feature(int(l)),
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


to_example(train_data, '../data/dedup/train.tfrecord')
to_example(test_data, '../data/dedup/test.tfrecord')
