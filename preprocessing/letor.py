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
        self.train_ftrs = Letor._exclude_not_found(train_ftrs)
        self.test_ftrs = Letor._exclude_not_found(test_ftrs)
        self.std_scaler_path = '../data/dedup/standard_scaler.pkl'
        self.value_cols = [c for c in test_ftrs.columns
                           if c not in INFO_COLUMNS]

    @staticmethod
    def _exclude_not_found(ftrs):
        """
        ############ RANKER: exclude groups with ix == -1 ###############
        Exclude whole groups from ranking because they don't have positive target
        """
        if ftrs is None:
            return None
        exclude = ftrs[ftrs['ix'] == -1]['synid'].unique()
        cond = ftrs['synid'].isin(exclude)
        return ftrs[~cond]

    def _split(self):
        if self.test_only:
            scaler = tools.do_unpickle(self.std_scaler_path)
            X_train, ixs_train, X_vali, ixs_vali = [None]*4
        else:
            self.train_ftrs.sort_values(['qid', 'synid'], inplace=True)

            qid_train, qid_vali = train_test_split(
                self.train_ftrs['qid'].unique(), test_size=0.1, random_state=42)
            cond = self.train_ftrs['qid'].isin(qid_train)
            train_part = self.train_ftrs[cond]
            vali_part = self.train_ftrs[~cond]
            ixs_train = train_part[INFO_COLUMNS].values
            ixs_vali = vali_part[INFO_COLUMNS].values

            scaler = StandardScaler()
            X_train = scaler.fit_transform(train_part[self.value_cols])
            X_vali = scaler.transform(vali_part[self.value_cols])
            tools.do_pickle(scaler, self.std_scaler_path)

        self.test_ftrs.sort_values(['qid', 'synid'], inplace=True)
        X_test = scaler.transform(self.test_ftrs[self.value_cols])
        ixs_test = self.test_ftrs[INFO_COLUMNS].values

        return X_train, ixs_train, X_vali, ixs_vali, X_test, ixs_test

    @staticmethod
    def letor_producer(X, ixs):
        _id = 0
        qid_prev, synid_prev = None, None
        for ids, row in tools.tqdm(zip(ixs, X), total=len(X)):
            qid, synid, fid, target = ids
            if (qid_prev, synid_prev) != (qid, synid):
                _id += 1
            qid_prev, synid_prev = qid, synid
            yield target, _id, row, ids

    def save_txt(self):
        X_train, ixs_train, X_vali, ixs_vali, X_test, ixs_test = self._split()

        self.to_txt(X_test, ixs_test, self.data_dir + '/test_letor.txt')
        if self.test_only:
            return
        self.to_txt(X_train, ixs_train, self.data_dir + '/train_letor.txt')
        self.to_txt(X_vali, ixs_vali, self.data_dir + '/vali_letor.txt')

    @staticmethod
    def to_txt(X, ixs, fname):
        gfile = fname.rstrip('txt') + 'group'
        ixfile = fname.rstrip('txt') + 'ix'
        with open(fname, 'w') as f, open(gfile, 'w') as g, open(ixfile, 'w') as ixf:
            gcount = 0
            prev_id = None
            for target, _id, row, ids in Letor.letor_producer(X, ixs):
                s = '%d qid:%d' % (target, _id)
                _sft = ' '.join(['%d:%f' % (i + 1, v)
                                 for i, v in enumerate(row)])
                s = ' '.join([s, _sft, '\n'])
                f.write(s)
                ixf.write('\t'.join(['%d' % i for i in ids])+'\n')

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
    def to_example(X, ixs, filename):
        X = X.astype(np.float32)
        writer = tf.python_io.TFRecordWriter(filename)
        _id_prev = None
        for target, _id, row, ids in Letor.letor_producer(X, ixs):
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
        X_train, ixs_train, X_vali, ixs_vali, X_test, ixs_test = self._split()

        self.to_example(X_test, ixs_test,
                        self.data_dir + '/test_letor.tfrecord')
        if self.test_only:
            return

        self.to_example(X_train, ixs_train,
                        self.data_dir + '/train_letor.tfrecord')
        self.to_example(X_vali, ixs_vali,
                        self.data_dir + '/vali_letor.tfrecord')
