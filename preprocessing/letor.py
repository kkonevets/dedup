import tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

INFO_COLUMNS = ['qid', 'synid', 'fid', 'target']


def letor_producer(X, qst):
    _id = 0
    qid_prev, synid_prev = None, None
    for (qid, synid, target), row in tools.tqdm(zip(qst, X), total=len(X)):
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


def save_letor_txt(train_sim_ftrs, test_sim_ftrs, data_dir, vali=False):
    X_train, qst_train, X_test, qst_test = letor_prepare(
        train_sim_ftrs, test_sim_ftrs)

    to_letor(X_test, qst_test, data_dir + '/test_letor.txt')

    if vali:
        hashtag = qst_train[:, :2]  # ['qid', 'synid']
        hashtag = pd.Series(map(tuple, hashtag))
        hash_train, hash_vali = train_test_split(
            hashtag.unique(), test_size=0.1, random_state=42)
        cond = hashtag.isin(hash_train).values
        to_letor(X_train[cond], qst_train[cond],
                 data_dir + '/train_letor.txt')
        to_letor(X_train[~cond], qst_train[~cond],
                 data_dir + '/vali_letor.txt')
    else:
        to_letor(X_train, qst_train, data_dir + '/train_letor.txt')


def _int32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def to_letor_example(train_sim_ftrs, test_sim_ftrs, data_dir):
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

    save_one(X_train, qst_train, data_dir + '/train_letor.tfrecord')
    save_one(X_test, qst_test, data_dir + '/test_letor.tfrecord')
