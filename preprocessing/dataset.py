import tools
import numpy as np
import tensorflow as tf
import os
import io
from preprocessing import textsim
from tqdm import tqdm


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

    labels = cur_samples['target'].values

    q_terms, d_terms, scores, ixs = [], [], [], []
    for row in cur_samples.itertuples():
        scores.append(row.score)
        ixs.append(row.ix)
        d_terms.append(fid2text[row.fid].split())
        if row.synid != -1:
            qtext = sid2text[row.synid]
        else:
            qtext = qid2text[row.qid]

        q_terms.append(qtext.split())

    # TODO: add DNN features: score, ix, brands ...

    return np.array(q_terms),  np.array(d_terms), scores, ixs, labels


train_data = input_data(True)
test_data = input_data(False)


def get_similarity_features(data):
    labels = data[2]
    columns = None
    vals = []
    for q_terms, d_terms, score, ix, label in tqdm(zip(*data), total=len(data[0])):
        q = ' '.join(q_terms)
        d = ' '.join(d_terms)
        ftrs = textsim.get_sim_features(q, d)
        vals.append(list(ftrs.values()) + [score, ix])
        if not columns:
            columns = list(ftrs.keys()) + ['score', 'ix']

    return vals, columns, labels


train_sim_ftrs = get_similarity_features(train_data)


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def to_example(data, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    q_terms, d_terms, scores, ixs, labels = data
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
