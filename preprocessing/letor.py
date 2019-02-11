import tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import preprocessing.tfrecord as tfrec

INFO_COLUMNS = ['qid', 'synid', 'fid', 'target']


class Letor:
    def __init__(self, data_dir, train_ftrs, test_ftrs):
        self.data_dir = data_dir
        self.test_only = train_ftrs is None
        self.qst_cols = ['qid', 'synid', 'target']
        self.train_ftrs = train_ftrs
        self.test_ftrs = test_ftrs
        self.std_scaler_path = '../data/dedup/phase1/standard_scaler.pkl'
        self.value_cols = [c for c in test_ftrs.columns
                           if c not in INFO_COLUMNS]

    def save_txt(self, vali=False):
        X_train, qst_train, X_test, qst_test = self._prepare()

        self.to_letor(X_test, qst_test, self.data_dir + '/test_letor.txt')
        if self.test_only:
            return

        if vali:
            hashtag = qst_train[:, : 2]  # ['qid', 'synid']
            hashtag = pd.Series(map(tuple, hashtag))
            hash_train, hash_vali = train_test_split(
                hashtag.unique(), test_size=0.1, random_state=42)
            cond = hashtag.isin(hash_train).values
            self.to_letor(X_train[cond], qst_train[cond],
                          self.data_dir + '/train_letor.txt')
            self.to_letor(X_train[~cond], qst_train[~cond],
                          self.data_dir + '/vali_letor.txt')
        else:
            self.to_letor(X_train, qst_train,
                          self.data_dir + '/train_letor.txt')

    def _prepare(self):
        if self.test_only:
            scaler = tools.do_unpickle(self.std_scaler_path)
            X_train, qst_train = None, None
        else:
            self.train_ftrs.sort_values(['qid', 'synid'], inplace=True)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(
                self.train_ftrs[self.value_cols])
            qst_train = self.train_ftrs[self.qst_cols].values
            tools.do_pickle(scaler, self.std_scaler_path)

        self.test_ftrs.sort_values(['qid', 'synid'], inplace=True)

        X_test = scaler.transform(self.test_ftrs[self.value_cols])
        qst_test = self.test_ftrs[self.qst_cols].values

        return X_train, qst_train, X_test, qst_test

    @staticmethod
    def letor_producer(X, qst):
        _id = 0
        qid_prev, synid_prev = None, None
        for (qid, synid, target), row in tools.tqdm(zip(qst, X), total=len(X)):
            if (qid_prev, synid_prev) != (qid, synid):
                _id += 1
            qid_prev, synid_prev = qid, synid
            yield target, _id, row

    @staticmethod
    def to_letor(X, qst, fname):
        gfile = fname.rstrip('txt') + 'group'
        with open(fname, 'w') as f, open(gfile, 'w') as g:
            gcount = 0
            prev_id = None
            for target, _id, row in Letor.letor_producer(X, qst):
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

    def to_letor_example(self, train_ftrs, test_ftrs):
        X_train, qst_train, X_test, qst_test = self._prepare()

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        def save_one(X, qst, filename):
            writer = tf.python_io.TFRecordWriter(filename)
            _id_prev = None
            for target, _id, row in self.letor_producer(X, qst):
                # Create a feature
                feature = {
                    'qid': tfrec._int32_feature(int(_id)),
                    'x': tf.train.Feature(float_list=tf.train.FloatList(value=row)),
                    'labels': tfrec._int32_feature(int(target)),
                }
                # Create an example protocol buffer
                example = tf.train.Example(
                    features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

            writer.close()

        save_one(X_train, qst_train, self.data_dir + '/train_letor.tfrecord')
        save_one(X_test, qst_test, self.data_dir + '/test_letor.tfrecord')
