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
from preprocessing.letor import save_letor_txt
from preprocessing.letor import to_letor_example

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


def main(argv):
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    prod = Producer(FLAGS.data_dir, COLNAMES)

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
        test_sim_ftrs = textsim.get_similarity_features(
            test_gen, COLNAMES, FLAGS.data_dir + '/test_sim_ftrs.npz')
        train_sim_ftrs = textsim.get_similarity_features(
            train_gen, COLNAMES, FLAGS.data_dir + '/train_sim_ftrs.npz')

    train_sim_ftrs, test_sim_ftrs = load_sim_ftrs(with_extra=False)
    save_letor_txt(train_sim_ftrs, test_sim_ftrs, FLAGS.data_dir, vali=True)

    # to_letor_example(train_sim_ftrs, test_sim_ftrs, FLAGS.data_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")

    if True:
        sys.argv += ['--data_dir=../data/dedup/phase2/', '--build_features']
        FLAGS(sys.argv)
    else:
        app.run(main)
