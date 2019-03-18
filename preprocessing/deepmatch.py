
from absl import flags
from absl import app
import tools
from tqdm import tqdm
import io
import os
import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from pymongo import MongoClient
from preprocessing.corpus import id2ets
import csv
import spacy
from functools import partial

FLAGS = tools.FLAGS

def frame_to_txt(df, mid2et, id2et, fname):
    # add score and ix as features
    columns = [
        'id','label',
        'left_name',# 'left_brand',
        'right_name',# 'right_brand',
    ]

    # nlp = spacy.load('xx')

    with io.open(fname, 'w', encoding='utf8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        for tup in tqdm(df.itertuples(), total=df.shape[0]):
            met = mid2et[tup.fid]
            sid = None if pd.isna(tup.synid) else tup.synid
            et = id2et[(tup.qid, sid)]
            row = [
                tup.Index, int(tup.target),
                et['text'],
                met['text']
            ]
            writer.writerow(row)


def generate_tts():
    directory = FLAGS.data_dir + '/deepmatch/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    samples = tools.load_samples(FLAGS.data_dir + '/samples.npz')
    samples = samples[samples['ix']!=-1]

    normalizer = partial(tools.normalize, stem=False,
                         translit=False, replace_i=False)
    mid2et, id2et = id2ets(samples, normalizer, all_master=False)

    random_state = 11

    train_samples = samples[samples['train']==1]
    test = samples[samples['train']==0].sample(frac=1, 
                        random_state=random_state)

    qids = train_samples['qid'].unique()

    qid_train, qid_vali = train_test_split(qids, test_size=0.1, 
                                        random_state=random_state)
    cond = train_samples['qid'].isin(qid_train)
    train = train_samples[cond].sample(frac=1,random_state=random_state)
    vali = train_samples[~cond].sample(frac=1,random_state=random_state)

    frame_to_txt(train, mid2et, id2et, directory + 'train.csv')
    frame_to_txt(vali, mid2et, id2et, directory + 'vali.csv')
    frame_to_txt(test, mid2et, id2et, directory + 'test.csv')


def main(argv):
    del argv  # Unused.


if __name__ == '__main__':
    import __main__
    flags.mark_flag_as_required("data_dir")

    if hasattr(__main__, '__file__'):
        app.run(main)
    else:
        sys.argv += ['--data_dir=../data/dedup',]
        FLAGS(sys.argv)
