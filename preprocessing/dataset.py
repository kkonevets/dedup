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
import io
import sys
from preprocessing import textsim
from tqdm import tqdm
import multiprocessing as mp
from itertools import islice
from functools import partial
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics.pairwise import pairwise_distances
from scipy.spatial.distance import cosine
from gensim.models import FastText
import nltk
import tensorflow as tf
from preprocessing.producing import Producer
from preprocessing.letor import Letor
from preprocessing.letor import INFO_COLUMNS
import preprocessing.tfrecord as tfrec


flags.DEFINE_string("data_dir", None, "path to data directory")
flags.DEFINE_bool("build_features", False, "build column features")
flags.DEFINE_bool("build_tfidf", False, "build tfidf features")
flags.DEFINE_bool("build_fasttext", False, "build fasttext features")
flags.DEFINE_bool("tfidf", False, "use tfidf features")
flags.DEFINE_bool("fasttext", False, "use fasttext features")
flags.DEFINE_bool("build_tfrecord", False,
                  "build tensorflow record input files")

FLAGS = flags.FLAGS
COLNAMES = INFO_COLUMNS + ['score', 'ix']


def to_example(data, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    q_terms, d_terms, info = data
    labels = info[:, -1]
    for q, d, l in zip(q_terms, d_terms, labels):
            # Create a feature
        feature = {
            'q_terms': tfrec._bytes_feature([tf.compat.as_bytes(qi) for qi in q]),
            'd_terms': tfrec._bytes_feature([tf.compat.as_bytes(di) for di in d]),
            'labels': tfrec._int32_feature(int(l)),
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()


def compute_tfidf_dists(train_gen, test_gen):
    tfidf_model = tools.do_unpickle('../data/dedup/tfidf_model.pkl')

    def get_dists(data, fname):
        qs, ds, ixs = [], [], []
        for q_terms, d_terms, _ixs in data:
            qs.append(' '.join(q_terms))
            ds.append(' '.join(d_terms))
            ixs.append(_ixs[:3])
        if len(qs) == 0:
            return
        qvecs = tfidf_model.transform(qs)
        dvecs = tfidf_model.transform(ds)

        dists = paired_cosine_distances(qvecs, dvecs)
        np.savez(fname, dists=np.hstack([ixs, np.array([dists]).T]))

    get_dists(train_gen, FLAGS.data_dir + '/train_tfidf_cosine.npz')
    get_dists(test_gen, FLAGS.data_dir + '/test_tfidf_cosine.npz')


def compute_fasttext_dists(train_gen_raw, test_gen_raw):
    model = FastText.load_fasttext_format(
        FLAGS.data_dir + '../vectors/cc.ru.300.bin')
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


def load_sim_ftrs():
    def dists_from_numpy(sim_ftrs, mname, tag):
        filename = FLAGS.data_dir + '/%s_%s_cosine.npz' % (tag, mname)
        df = np.load(filename)['dists']
        df = pd.DataFrame(df)
        columns = ['qid', 'synid', 'fid']
        columns += ['%s%d' % (mname, i) for i in range(df.shape[1]-3)]
        df.columns = columns
        df.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
        return sim_ftrs.merge(df, on=['qid', 'synid', 'fid'])

    def load_one(train=True):
        tag = 'train' if train else 'test'
        filename = FLAGS.data_dir + '/%s_sim_ftrs.npz' % tag
        sim_ftrs = tools.load_samples(filename, key='vals')
        if sim_ftrs is None:
            return

        sim_ftrs.drop_duplicates(['qid', 'synid', 'fid'], inplace=True)
        if FLAGS.tfidf:
            sim_ftrs = dists_from_numpy(sim_ftrs, 'tfidf', tag)
        if FLAGS.fasttext:
            sim_ftrs = dists_from_numpy(sim_ftrs, 'fasttext', tag)
        sim_ftrs.fillna(-1, inplace=True)
        return sim_ftrs

    test_sim_ftrs = load_one(False)
    train_sim_ftrs = load_one(True)

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
        test_sim_ftrs = textsim.get_similarity_features(
            test_gen, COLNAMES, FLAGS.data_dir + '/test_sim_ftrs.npz')
        train_sim_ftrs = textsim.get_similarity_features(
            train_gen, COLNAMES, FLAGS.data_dir + '/train_sim_ftrs.npz')

    train_sim_ftrs, test_sim_ftrs = load_sim_ftrs()
    letor = Letor(FLAGS.data_dir, train_sim_ftrs, test_sim_ftrs)
    letor.save_txt()


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")

    if True:
        sys.argv += ['--data_dir=../data/dedup',
                     '--build_features', '--build_tfidf', '--tfidf']
        FLAGS(sys.argv)
    else:
        app.run(main)
