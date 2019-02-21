r"""
Sample command lines:

python3 simboost.py \
--data_dir=../data/dedup/ \
--tfidf

"""

from absl import flags
from absl import app
import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd
from scipy import sparse
import shutil
import os
import tools
from preprocessing.dataset import load_sim_ftrs
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from preprocessing.letor import INFO_COLUMNS
import sys
import scoring
from pymongo import MongoClient

FLAGS = tools.FLAGS


def get_groups(fname):
    groups = []
    with open(fname, "r") as f:
        data = f.readlines()
        for line in data:
            groups.append(int(line.split("\n")[0]))
    return groups


def build_ranker():
    dtrain = xgb.DMatrix(FLAGS.data_dir + 'train_letor.txt')
    dvali = xgb.DMatrix(FLAGS.data_dir + 'vali_letor.txt')
    dtest = xgb.DMatrix(FLAGS.data_dir + 'test_letor.txt')

    group_train = get_groups(FLAGS.data_dir + '/train_letor.group')
    group_vali = get_groups(FLAGS.data_dir + '/vali_letor.group')
    group_test = get_groups(FLAGS.data_dir + '/test_letor.group')

    recall_scale = scoring.get_recall_test_scale()

    params = {
        'objective': 'rank:ndcg',
        'max_depth': 10,
        'eta': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'eval_metric': ['ndcg@2', 'map@2', 'ndcg@1']
    }
    xgb_ranker = xgb.train(params, dtrain,
                           num_boost_round=5000,
                           early_stopping_rounds=20,
                           evals=[(dvali, 'vali')])

    ranks = xgb_ranker.predict(dtest)
    scores, positions = scoring.ranker_predict(ranks, dtest, group_test)
    tools.pprint(scores)
    # print(xgb_ranker.eval(dvali))

    joblib.dump(xgb_ranker, FLAGS.data_dir + '/xgb_ranker.model')
    # xgb_ranker = joblib.load('../data/dedup/xgb_ranker.model')

    itest = pd.read_csv('../data/dedup/test_letor.ix',
                        header=None, sep='\t')
    itest.columns = ['qid', 'synid', 'fid', 'target']

    client = MongoClient(tools.c_HOST)
    db = client['cache']
    test_qids = itest['qid'].unique().tolist()
    positions_solr = db['solr_positions'].find(
        {'et_id': {'$in': test_qids}, 'i': {'$lte': max(group_test)-1}}, projection=['i'])
    positions_solr = pd.Series([p['i'] for p in positions_solr if p['i'] >= 0])

    scoring.plot_topn_curves([positions, positions_solr],
                             '../data/dedup/cumsum_test.pdf', scale=recall_scale,
                             labels=['reranking', 'SOLR'], title='Test: found in top N')


def build_classifier():
    ftrain, ftest = load_sim_ftrs()
    ftrain = ftrain[ftrain['ix'] != -1]
    ftest = ftest[ftest['ix'] != -1]

    recall_scale = scoring.get_recall_test_scale()

    qid_train, _ = train_test_split(
        ftrain['qid'].unique(), test_size=0.1, random_state=42)

    cond = ftrain['qid'].isin(qid_train)
    train_part = ftrain[cond]
    vali_part = ftrain[~cond]

    value_cols = [c for c in ftrain.columns if c not in INFO_COLUMNS]

    dtrain = DMatrix(train_part[value_cols], label=train_part['target'])
    dvali = DMatrix(vali_part[value_cols], label=vali_part['target'])
    dtest = DMatrix(ftest[value_cols], label=ftest['target'])

    params = {
        'objective': 'binary:logistic',
        'max_depth': 10,  # 10 best
        'eval_metric': ['logloss'],
        'eta': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1
    }

    xgb_clr = xgb.train(params, dtrain,
                        num_boost_round=1000,
                        early_stopping_rounds=10,
                        evals=[(dvali, 'vali')])

    train_probs = xgb_clr.predict(dtrain)
    _ = scoring.clr_predict(train_probs, dtrain)

    test_probs = xgb_clr.predict(dtest)
    y_pred = scoring.clr_predict(test_probs, dtest, threshold=0.5)
    scoring.plot_precision_recall(dtest.get_label(), test_probs, tag='',
                                  recall_scale=recall_scale)
    cm = confusion_matrix(dtest.get_label(), y_pred)
    print(cm)

    joblib.dump(xgb_clr, FLAGS.data_dir + '/xgb_clr.model')
    # xgb_clr = joblib.load('../data/dedup/xgb_clr.model')

    # for max_depth in [3, 5, 8, 10, 15]:
    #     params['max_depth'] = max_depth
    #     xgb_clr = xgb.train(params, dtrain,
    #                         num_boost_round=1000,
    #                         early_stopping_rounds=10,
    #                         evals=[(dvali, 'vali')])
    #     y_pred = clr_predict(xgb_clr, dtest, threshold=0.5, tag=str(max_depth))

    # ftest['prob'] = test_probs
    # ftest[INFO_COLUMNS+['score', 'prob']].sort_values(['qid', 'synid', 'fid', 'target'])

    scoring.examples_to_view(
        ftest, test_probs, FLAGS.feed_db, FLAGS.release_db)
    scoring.plot_binary_prob_freqs(dtest.get_label(), test_probs)


def test():
    xgb_clr = joblib.load('../data/dedup/xgb_clr.model')
    ftrain, ftest = load_sim_ftrs()
    value_cols = [c for c in ftest.columns if c not in INFO_COLUMNS]
    dtest = DMatrix(ftest[value_cols])
    probs = xgb_clr.predict(dtest)

    ftest['prob'] = probs
    sub = ftest[['qid', 'synid', 'fid', 'score', 'ix', 'prob']]

    maxs = sub[['qid', 'prob']].groupby('qid').max()['prob']
    N = len(sub['ix'].unique())
    ax = maxs.hist()
    fig = ax.get_figure()
    ax.set_xlabel("excluded probs")
    ax.set_ylabel("frequency")
    fig.savefig('../data/dedup/notexisting_probs.pdf')

    # sub[sub['prob'] > 0.9].to_excel('../data/dedup/samples_look.xlsx', index=False)


def main(argv):
    build_ranker()
    build_classifier()
    # test()


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")

    if False:
        sys.argv += ['--data_dir=../data/dedup/', '--tfidf']
        FLAGS(sys.argv)
    else:
        app.run(main)
