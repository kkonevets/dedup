
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
from functools import partial

FLAGS = tools.FLAGS

def frame_to_txt(df, mid2et, id2et, fname):
    # add score and ix as features
    columns = [
        'id','label',
        'left_name',# 'left_brand',
        'right_name',# 'right_brand',
    ]

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


def print_target_split(df, tag):
    df = df[['qid', 'synid', 'target']].copy()
    df.fillna(-1, inplace=True)
    grouped = df.groupby(['qid', 'synid'], sort=False)
    qst = grouped.apply(lambda g: g['target'].max())
    vals = qst.value_counts().to_dict()
    print('%s: %f'% (tag, vals[1]/(vals[1]+vals[0])))


def generate_tts():
    directory = FLAGS.data_dir + '/deepmatch/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    samples = tools.load_samples(FLAGS.data_dir + '/samples.npz')
    samples = samples[samples['ix']!=-1]

    random_state = 11

    train = samples[(samples['train']==1)&(samples['vali']==0)]
    train = train.sample(frac=1,random_state=random_state)
    vali = samples[samples['vali']==1].sample(frac=1,random_state=random_state)
    test = samples[samples['train']==0]

    print_target_split(samples, 'samples')
    print_target_split(train, 'train')
    print_target_split(vali, 'vali')
    print_target_split(test, 'test')

    normalizer = partial(tools.normalize, stem=False,
                         translit=False, replace_i=False)
    mid2et, id2et = id2ets(samples, normalizer, all_master=False)

    frame_to_txt(train, mid2et, id2et, directory + 'train.csv')
    frame_to_txt(vali, mid2et, id2et, directory + 'vali.csv')
    frame_to_txt(test, mid2et, id2et, directory + 'test.csv')

    # train_orgs = samples[(samples['train']==1)&(samples['vali']==0)]['org'].unique()
    # test_orgs = samples[samples['train']==0]['org'].unique()
    # vali_orgs = samples[samples['vali']==1]['org'].unique()

    # set(train_orgs).intersection(test_orgs)
    # set(train_orgs).intersection(vali_orgs)

def main(argv):
    del argv  # Unused.
    generate_tts()


if __name__ == '__main__':
    import __main__
    flags.mark_flag_as_required("data_dir")

    if hasattr(__main__, '__file__'):
        app.run(main)
    else:
        sys.argv += ['--data_dir=../data/dedup',]
        FLAGS(sys.argv)
