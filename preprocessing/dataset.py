r"""
Sample command lines:

python3 preprocessing/dataset.py \
--data_dir=../data/dedup/phase2/ \
--build_features \
--ftidf

"""

from absl import flags
from absl import app
import tools
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import io
import sys
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
import nltk

flags.DEFINE_string("data_dir", None, "path to data directory")
flags.DEFINE_bool("build_features", False, "build column features")
flags.DEFINE_bool("build_tfidf", False, "build tfidf features")
flags.DEFINE_bool("build_fasttext", False, "build fasttext features")
flags.DEFINE_bool("tfidf", False, "use tfidf features")
flags.DEFINE_bool("fasttext", False, "use fasttext features")
flags.DEFINE_bool("build_tfrecord", False,
                  "build tensorflow record input files")


FLAGS = flags.FLAGS
INFO_COLUMNS = ['qid', 'synid', 'fid', 'target']
COLNAMES = INFO_COLUMNS + ['score', 'ix']


def sim_worker(tup):
    q_terms, d_terms, info = tup
    ftrs = textsim.get_sim_features(q_terms, d_terms)
    values = list(info) + list(ftrs.values())
    columns = COLNAMES + list(ftrs.keys())
    return values, columns


def get_similarity_features(data_gen, output_file):
    columns = None

    vals = []
    with mp.Pool(mp.cpu_count(), maxtasksperchild=100000) as p:
        for values, columns in p.imap_unordered(sim_worker, data_gen):
            vals.append(values)

    # for values, columns in map(wraper, data_gen):
    #     vals.append(values)

    vals = np.array(vals, dtype=np.float32)
    np.savez(output_file, vals=vals, columns=columns)
    return vals, columns


def compute_tfidf_dists(train_gen, test_gen):
    tfidf_model = tools.do_unpickle(FLAGS.data_dir + '/tfidf_model.pkl')

    def get_dists(data, fname):
        qs = [' '.join(q_terms) for q_terms in data[0]]
        ds = [' '.join(d_terms) for d_terms in data[1]]
        qvecs = tfidf_model.transform(qs)
        dvecs = tfidf_model.transform(ds)

        dists = paired_cosine_distances(qvecs, dvecs)
        ixs = data[2][:, :3]
        np.savez(fname, dists=np.hstack([ixs, np.array([dists]).T]))

    get_dists(train_gen, FLAGS.data_dir + '/train_tfidf_cosine.npz')
    get_dists(test_gen, FLAGS.data_dir + '/test_tfidf_cosine.npz')


def compute_fasttext_dists(train_gen_raw, test_gen_raw):
    model = FastText.load_fasttext_format(FLAGS.data_dir + '/cc.ru.300.bin')
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

    get_dists(train_gen_raw, FLAGS.data_dir + '/train_fasttext_cosine.npz')
    get_dists(test_gen_raw, FLAGS.data_dir + '/test_fasttext_cosine.npz')


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
    gfile = fname.rstrip('txt') + 'group'
    with open(fname, 'w') as f, open(gfile, 'w') as g:
        gcount = 0
        prev_id = None
        for target, _id, row in letor_producer(X, qst):
            s = '%d qid:%d' % (target, _id)
            _sft = ' '.join(['%d:%f' % (i + 1, v)
                             for i, v in enumerate(row)])
            s = ' '.join([s, _sft, '\n'])
            f.write(s)

            if prev_id is None:
                prev_id = _id
            else:
                gcount += 1
            if _id != prev_id:
                g.write('%d\n' % gcount)
                gcount = 0
            prev_id = _id
        g.write('%d\n' % (gcount + 1))


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

    to_letor(X_test, qst_test, FLAGS.data_dir + '/test_letor.txt')

    if vali:
        hashtag = qst_train[:, :2]  # ['qid', 'synid']
        hashtag = pd.Series(map(tuple, hashtag))
        hash_train, hash_vali = train_test_split(
            hashtag.unique(), test_size=0.1, random_state=42)
        cond = hashtag.isin(hash_train).values
        to_letor(X_train[cond], qst_train[cond],
                 FLAGS.data_dir + '/train_letor.txt')
        to_letor(X_train[~cond], qst_train[~cond],
                 FLAGS.data_dir + '/vali_letor.txt')
    else:
        to_letor(X_train, qst_train, FLAGS.data_dir + '/train_letor.txt')


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

    save_one(X_train, qst_train, FLAGS.data_dir + '/train_letor.tfrecord')
    save_one(X_test, qst_test, FLAGS.data_dir + '/test_letor.tfrecord')


def load_sim_ftrs(with_extra=False):
    def dists_from_numpy(fname, tag):
        df = np.load(fname)['dists']
        df = pd.DataFrame(df)
        columns = ['qid', 'synid', 'fid']
        columns += ['%s%d' % (tag, i) for i in range(df.shape[1]-3)]
        df.columns = columns
        df.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
        return df

    test_sim_ftrs = tools.load_samples(
        FLAGS.data_dir + '/test_sim_ftrs.npz', key='vals')
    train_sim_ftrs = tools.load_samples(
        FLAGS.data_dir + '/train_sim_ftrs.npz', key='vals')

    test_sim_ftrs.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
    train_sim_ftrs.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)

    test_tfidf_cos = dists_from_numpy(
        FLAGS.data_dir + '/test_tfidf_cosine.npz', 'tfidf')
    train_tfidf_cos = dists_from_numpy(
        FLAGS.data_dir + '/train_tfidf_cosine.npz', 'tfidf')

    test_sim_ftrs = test_sim_ftrs.merge(
        test_tfidf_cos, on=['qid', 'synid', 'fid'])
    train_sim_ftrs = train_sim_ftrs.merge(
        train_tfidf_cos, on=['qid', 'synid', 'fid'])

    # test_ft_cos = dists_from_numpy(
    #     FLAGS.data_dir + '/test_fasttext_cosine.npz', 'ft')
    # train_ft_cos = dists_from_numpy(
    #     FLAGS.data_dir + '/train_fasttext_cosine.npz', 'ft')

    # test_sim_ftrs = test_sim_ftrs.merge(
    #     test_ft_cos, on=['qid', 'synid', 'fid'])
    # train_sim_ftrs = train_sim_ftrs.merge(
    #     train_ft_cos, on=['qid', 'synid', 'fid'])

    test_sim_ftrs.fillna(-1, inplace=True)
    train_sim_ftrs.fillna(-1, inplace=True)

    if with_extra:
        print("WARNING: Loading extra features!")
        test_extra = tools.load_samples(
            FLAGS.data_dir + '/test_sim_ftrs_extra.npz', key='vals')
        usecols = ['qid', 'synid', 'fid'] + \
            list(set(test_extra.columns).difference(COLNAMES))
        test_extra = test_extra[usecols]
        train_extra = tools.load_samples(
            FLAGS.data_dir + '/train_sim_ftrs_extra.npz', key='vals')
        train_extra = train_extra[usecols]

        # unite with extra
        test_sim_ftrs = test_sim_ftrs.merge(
            test_extra, on=['qid', 'synid', 'fid'])
        train_sim_ftrs = train_sim_ftrs.merge(
            train_extra, on=['qid', 'synid', 'fid'])

    return train_sim_ftrs, test_sim_ftrs


class Producer:
    def __init__(self):
        self.load_data()

    def gen_pairs(self):
        samples, qid2text, sid2text, fid2text = \
            self.samples, self.qid2text, self.sid2text, self.fid2text
        if 'train' in samples.columns:
            train_gen = self.gen_data(
                samples[samples['train'] == 1])
            test_samples = samples[samples['train'] == 0]
        else:
            train_gen = iter(())
            test_samples = samples

        test_gen = self.gen_data(test_samples)
        return train_gen, test_gen

    @staticmethod
    def get_id2text(corpus, tag, ids):
        id2text = {}
        for _id, text in corpus[corpus[tag].isin(ids)][[tag, 'text']].values:
            id2text[_id] = text
        return id2text

    def gen_data(self, cur_samples):
        fid2text, sid2text, qid2text = self.fid2text, self.sid2text, self.qid2text
        for row in tqdm(cur_samples.itertuples(), total=len(cur_samples)):
            d_splited = fid2text[row.fid].split()
            if row.synid != -1:
                qtext = sid2text[row.synid]
            else:
                qtext = qid2text[row.qid]

            q_splited = qtext.split()
            if row.target == 0 and ' '.join(d_splited) == ' '.join(q_splited):
                continue

            if len(q_splited) * len(d_splited) == 0:
                continue

            # TODO: add DNN features: brands ...
            yield q_splited, d_splited, [getattr(row, c) for c in COLNAMES]

    def load_data(self):
        samples = tools.load_samples(FLAGS.data_dir + '/samples.npz')

        # exclude samples not found in TOP
        synids_exclude = set(samples[samples['ix'] == -1]['synid'].unique())
        synids_exclude.discard(-1)
        samples = samples[~samples['synid'].isin(synids_exclude)]
        qids_exclude = samples[samples['ix'] == -1]['qid'].unique()
        samples = samples[~samples['qid'].isin(qids_exclude)]

        qids = samples[samples['synid'] == -1]['qid'].unique()
        sids = samples[samples['synid'] != -1]['synid'].unique()
        fids = samples['fid'].unique()

        corpus = tools.load_samples(FLAGS.data_dir + '/corpus.npz')

        qid2text = self.get_id2text(corpus, 'qid', qids)
        sid2text = self.get_id2text(corpus, 'synid', sids)
        fid2text = self.get_id2text(corpus, 'fid', fids)

        # vals = corpus[corpus['train'] != 0]['text'].values
        # informative_terms = set([w for s in vals for w in s.split()])
        # with io.open(FLAGS.data_dir + '/vocab.txt', 'w', encoding='utf8') as f:
        #     for term in informative_terms:
        #         f.write(term + '\n')

        self.samples, self.qid2text, self.sid2text, self.fid2text = \
            samples, qid2text, sid2text, fid2text


def main(argv):
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    prod = Producer()

    if FLAGS.build_features:
        if FLAGS.build_tfidf:
            compute_tfidf_dists(*prod.gen_pairs())
        if FLAGS.build_fasttext:
            # do not pass traslited words
            compute_fasttext_dists(*prod.gen_pairs())
        if FLAGS.build_tfrecord:
            train_gen, test_gen = prod.gen_pairs()
            to_example(train_gen, FLAGS.data_dir + '/train.tfrecord')
            to_example(test_gen, FLAGS.data_dir + '/test.tfrecord')

        train_gen, test_gen = prod.gen_pairs()
        # data_gen, output_file = test_gen, FLAGS.data_dir + '/test_sim_ftrs.npz'
        test_sim_ftrs = get_similarity_features(
            test_gen, FLAGS.data_dir + '/test_sim_ftrs.npz')
        train_sim_ftrs = get_similarity_features(
            train_gen, FLAGS.data_dir + '/train_sim_ftrs.npz')

    train_sim_ftrs, test_sim_ftrs = load_sim_ftrs(with_extra=False)
    save_letor_txt(train_sim_ftrs, test_sim_ftrs, vali=True)

    # to_letor_example(train_sim_ftrs, test_sim_ftrs)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")

    if True:
        sys.argv += ['--data_dir=../data/dedup/phase2/', '--build_features']
        FLAGS(sys.argv)
    else:
        app.run(main)
