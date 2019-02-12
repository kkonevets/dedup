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
        self.std_scaler_path = '../data/dedup/standard_scaler.pkl'
        self.value_cols = [c for c in test_ftrs.columns
                           if c not in INFO_COLUMNS]

    def _split(self):
        if self.test_only:
            scaler = tools.do_unpickle(self.std_scaler_path)
            X_train, qst_train, X_vali, qst_vali = [None]*4
        else:
            self.train_ftrs.sort_values(['qid', 'synid'], inplace=True)

            qid_train, qid_vali = train_test_split(
                self.train_ftrs['qid'].unique(), test_size=0.1, random_state=42)
            cond = self.train_ftrs['qid'].isin(qid_train)
            train_part = self.train_ftrs[cond]
            vali_part = self.train_ftrs[~cond]
            qst_train = train_part[self.qst_cols].values
            qst_vali = vali_part[self.qst_cols].values

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_part[self.value_cols])
            X_vali = scaler.transform(vali_part[self.value_cols])
            tools.do_pickle(scaler, self.std_scaler_path)

        self.test_ftrs.sort_values(['qid', 'synid'], inplace=True)
        X_test = scaler.transform(self.test_ftrs[self.value_cols])
        qst_test = self.test_ftrs[self.qst_cols].values

        return X_train, qst_train, X_vali, qst_vali, X_test, qst_test

    @staticmethod
    def letor_producer(X, qst):
        _id = 0
        qid_prev, synid_prev = None, None
        for (qid, synid, target), row in tools.tqdm(zip(qst, X), total=len(X)):
            if (qid_prev, synid_prev) != (qid, synid):
                _id += 1
            qid_prev, synid_prev = qid, synid
            yield target, _id, row

    def save_txt(self):
        X_train, qst_train, X_vali, qst_vali, X_test, qst_test = self._split()

        self.to_txt(X_test, qst_test, self.data_dir + '/test_letor.txt')
        if self.test_only:
            return
        self.to_txt(X_train, qst_train, self.data_dir + '/train_letor.txt')
        self.to_txt(X_vali, qst_vali, self.data_dir + '/vali_letor.txt')

    @staticmethod
    def to_txt(X, qst, fname):
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

    @staticmethod
    def to_example(X, qst, filename):
        X = X.astype(np.float32)
        writer = tf.python_io.TFRecordWriter(filename)
        _id_prev = None
        for target, _id, row in Letor.letor_producer(X, qst):
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

    def save_example(self):
        X_train, qst_train, X_vali, qst_vali, X_test, qst_test = self._split()

        self.to_example(X_test, qst_test,
                        self.data_dir + '/test_letor.tfrecord')
        if self.test_only:
            return

        self.to_example(X_train, qst_train,
                        self.data_dir + '/train_letor.tfrecord')
        self.to_example(X_vali, qst_vali,
                        self.data_dir + '/vali_letor.tfrecord')
