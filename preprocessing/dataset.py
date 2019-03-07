r"""
Sample command lines:

python3 preprocessing/dataset.py \
--data_dir=../data/dedup \
--build_features \
--build_tfidf \
--tfidf

"""

from absl import flags
from absl import app
import tools
import numpy as np
import pandas as pd
import os
import sys
import h5py
from preprocessing import textsim
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine
from gensim.models import FastText
import tensorflow as tf
from preprocessing.producing import Producer
from preprocessing.letor import Letor
from preprocessing.letor import INFO_COLUMNS
import preprocessing.tfrecord as tfrec
from functools import lru_cache

FLAGS = tools.FLAGS

COLNAMES = INFO_COLUMNS + ['score', 'ix']


def to_example(gen, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for qdi in gen:
            # Create a feature
        feature = {
            'q_terms': tfrec._bytes_feature([tf.compat.as_bytes(qi) for qi in qdi.q_terms]),
            'd_terms': tfrec._bytes_feature([tf.compat.as_bytes(di) for di in qdi.d_terms]),
            'labels': tfrec._int32_feature(int(qdi.ix[3])),
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def compute_tfidf_dists(train_gen, test_gen):
    model = tools.do_unpickle('../data/dedup/tfidf_model.pkl')

    def get_dists(gen, fname):
        qs, ds, ixs = [], [], []
        for qdi in gen:
            qs.append(' '.join(qdi.q_terms))
            ds.append(' '.join(qdi.d_terms))
            ixs.append(qdi.ixs[:3])
        if len(qs) == 0:
            return
        qvecs = model.transform(qs)
        dvecs = model.transform(ds)

        dists = paired_cosine_distances(qvecs, dvecs)
        np.savez(fname, vals=np.hstack([ixs, np.array([dists]).T]))

    get_dists(train_gen, FLAGS.data_dir + '/train_tfidf_cosine.npz')
    get_dists(test_gen, FLAGS.data_dir + '/test_tfidf_cosine.npz')


def compute_fasttext_dists(train_gen_raw, test_gen_raw):
    from gensim.models import FastText

    # model = FastText.load_fasttext_format(
    #     FLAGS.data_dir + '../vectors/cc.ru.300.bin')
    model = FastText.load(FLAGS.data_dir + '/ftext.model')

    def get_dists(gen, fname):
        @lru_cache(maxsize=1)
        def get_prev(terms):
            vecs = [model.wv[t] for t in tools.replace_num(terms)
                    if t in model.wv]
            mean = np.mean(vecs, axis=0)
            return vecs, mean

        dists = []
        for qdi in gen:
            qvecs, qmean = get_prev(tuple(qdi.q_terms))
            dvecs = [model.wv[term] for term in tools.replace_num(qdi.d_terms) if
                     term in model.wv]
            dmean = np.mean(dvecs, axis=0)
            if len(qvecs) and len(dvecs):
                paired = pairwise_distances(qvecs, dvecs, metric='cosine')
                mins = np.min(paired, axis=1)
                mean, median, std = np.mean(mins), \
                    np.median(mins), np.std(mins)
            else:
                mean, median, std = -1, -1, -1
            dists.append(list(qdi.ixs[:3]) + [cosine(qmean, dmean),
                                              mean, median, std])
        dists = np.array(dists)
        dists[:, 3:] = dists[:, 3:].astype(np.float32)
        np.savez(fname, vals=dists)

    get_dists(train_gen_raw, FLAGS.data_dir + '/train_fasttext_cosine.npz')
    get_dists(test_gen_raw, FLAGS.data_dir + '/test_fasttext_cosine.npz')


def load_sim_ftrs():
    def dists_from_numpy(sim_ftrs, mname, filename):
        df = np.load(filename)['vals']
        df = pd.DataFrame(df)
        columns = ['qid', 'synid', 'fid']
        columns += ['%s%d' % (mname, i) for i in range(df.shape[1]-3)]
        df.columns = columns
        df.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
        return sim_ftrs.merge(df, on=['qid', 'synid', 'fid'])

    def load_one(tag='train'):
        filename = FLAGS.data_dir + '/%s_sim_ftrs.h5' % tag
        if not os.path.isfile(filename):
            return
        with h5py.File(filename, 'r') as hf:
            sim_ftrs = pd.DataFrame(hf['ftrs'][:])
            sim_ftrs.columns = hf['ftrs'].attrs['columns']

        sim_ftrs.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
        if FLAGS.tfidf:
            filename = FLAGS.data_dir + '/%s_tfidf_cosine.npz' % (tag)
            sim_ftrs = dists_from_numpy(sim_ftrs, 'tfidf', filename)
        if FLAGS.fasttext:
            filename = FLAGS.data_dir + '/%s_fasttext_cosine.npz' % (tag)
            sim_ftrs = dists_from_numpy(sim_ftrs, 'fasttext', filename)

        sim_ftrs.fillna(-1, inplace=True)
        return sim_ftrs

    test_sim_ftrs = load_one('test')
    train_sim_ftrs = load_one('train')

    return train_sim_ftrs, test_sim_ftrs


def main(argv):
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    prod = Producer(FLAGS.data_dir, COLNAMES)

    if FLAGS.build_tfidf:
        compute_tfidf_dists(*prod.gen_pairs())
    if FLAGS.build_fasttext:
        # do not pass traslited words
        compute_fasttext_dists(*prod.gen_pairs())
    if FLAGS.build_tfrecord:
        train_gen, test_gen = prod.gen_pairs()
        to_example(train_gen, FLAGS.data_dir + '/train.tfrecord')
        to_example(test_gen, FLAGS.data_dir + '/test.tfrecord')
    if FLAGS.build_features:
        train_gen, test_gen = prod.gen_pairs()
        textsim.extract_similarity_features(
            test_gen, COLNAMES, FLAGS.data_dir + '/test_sim_ftrs.h5')
        textsim.extract_similarity_features(
            train_gen, COLNAMES, FLAGS.data_dir + '/train_sim_ftrs.h5')

    train_sim_ftrs, test_sim_ftrs = load_sim_ftrs()
    letor = Letor(FLAGS.data_dir, train_sim_ftrs, test_sim_ftrs)
    letor.save_txt()


if __name__ == '__main__':
    import __main__
    flags.mark_flag_as_required("data_dir")

    if hasattr(__main__, '__file__'):
        app.run(main)
    else:
        sys.argv += ['--data_dir=../data/dedup',
                     '--build_features', '--build_tfidf', '--tfidf',
                     '--build_fasttext', '--fasttext']
        FLAGS(sys.argv)
