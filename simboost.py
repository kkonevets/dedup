r"""
Sample command lines:

python3 preprocessing/simboost.py \
--data_dir=../data/dedup/phase1/ \

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
from preprocessing import dataset
from preprocessing.textsim import get_sim_features
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import sys

FLAGS = flags.FLAGS


def ranker_predict(model, dmatrix, groups):
    ranks = model.predict(dmatrix)
    y = dmatrix.get_label()
    ix_prev = 0
    scores = {i: [] for i in range(1, 7)}
    for gcount in tools.tqdm(groups):
        y_cur = y[ix_prev: ix_prev + gcount]
        r = ranks[ix_prev: ix_prev + gcount]
        rsorted = y_cur[np.argsort(r)[::-1]]
        for k in scores.keys():
            val = tools.ndcg_at_k(rsorted, k, method=1)
            scores[k].append(val)
        ix_prev += gcount

    for k in list(scores.keys()):
        scores['ndcg@%d' % k] = np.round(np.mean(scores.pop(k)), 4)

    return scores


def clr_predict(model, dmtx, threshold=0.4):
    y = dmtx.get_label()
    c = Counter(y)
    probs = model.predict(dmtx)
    y_pred = (probs >= threshold).astype(int)
    rep = classification_report(y, y_pred, labels=[1], output_dict=True)
    rep = rep['1']
    rep['base_accuracy'] = c[0]/sum(c.values())
    rep['accuracy'] = accuracy_score(y, y_pred)
    rep = {k: round(v, 4) for k, v in rep.items()}
    tools.pprint(rep)
    print('\n')
    return y_pred


def build_ranker():
    dtrain = xgb.DMatrix(FLAGS.data_dir + 'train_letor.txt')
    dvali = xgb.DMatrix(FLAGS.data_dir + 'vali_letor.txt')
    dtest = xgb.DMatrix(FLAGS.data_dir + 'test_letor.txt')

    def get_groups(fname):
        groups = []
        with open(fname, "r") as f:
            data = f.readlines()
            for line in data:
                groups.append(int(line.split("\n")[0]))
        return groups

    group_train = get_groups(FLAGS.data_dir + '/train_letor.group')
    group_vali = get_groups(FLAGS.data_dir + '/vali_letor.group')
    group_test = get_groups(FLAGS.data_dir + '/test_letor.group')

    params = {
        'objective': 'rank:ndcg',
        'eta': 0.1,
        'max_depth': 10,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'eval_metric': ['ndcg@1', 'ndcg@2', 'map@2']
    }
    xgb_ranker = xgb.train(params, dtrain,
                           num_boost_round=1000,
                           early_stopping_rounds=20,
                           evals=[(dvali, 'vali')])

    tools.pprint(ranker_predict(xgb_ranker, dtest, group_test))
    # print(xgb_ranker.eval(dvali))

    joblib.dump(xgb_ranker, FLAGS.data_dir + '/xgb_ranker.model')

    # np.savez('../data/dedup/ranks.npz',
    #          train=xgb_ranker.predict(dtrain),
    #          vali=xgb_ranker.predict(dvali),
    #          test=xgb_ranker.predict(dtest))


def build_classifier():
    dtrain = xgb.DMatrix(FLAGS.data_dir + 'train_letor.txt')
    dvali = xgb.DMatrix(FLAGS.data_dir + 'vali_letor.txt')
    dtest = xgb.DMatrix(FLAGS.data_dir + 'test_letor.txt')

    # ranks = np.load('../data/dedup/ranks.npz')
    # rank_train = np.reshape(ranks['train'], (-1, 1))
    # rank_vali = np.reshape(ranks['vali'], (-1, 1))
    # rank_test = np.reshape(ranks['test'], (-1, 1))

    # X_train = sparse.hstack([X_train, rank_train])
    # X_vali = sparse.hstack([X_vali, rank_vali])
    # X_test = sparse.hstack([X_test, rank_test])

    params = {
        'objective': 'binary:logistic',
        'max_depth': 10,  # 10 best
        'learning_rate': 0.1,  # !!!!!!!!!!!!!!!
        'eval_metric': ['logloss']
        #   'min_child_weight': 1,
        #   'gamma': 3,
        #   'subsample': 0.8,
        #   'colsample_bytree': 0.8,
        #   'reg_alpha': 5
    }

    xgb_clr = xgb.train(params, dtrain,
                        num_boost_round=1000,
                        early_stopping_rounds=20,
                        evals=[(dvali, 'vali')])

    _ = clr_predict(xgb_clr, dtrain)
    y_pred = clr_predict(xgb_clr, dtest)
    cm = confusion_matrix(dtest.get_label(), y_pred)
    print(cm)

    joblib.dump(xgb_clr, FLAGS.data_dir + '/xgb_clr.model')


def test():
    dtest = xgb.DMatrix('../data/dedup/phase1/test_letor.txt')
    group_test = get_groups(FLAGS.data_dir + '/test_letor.group')

    xgb_ranker = joblib.load('../data/dedup/phase1/xgb_ranker.model')
    xgb_clr = joblib.load('../data/dedup/phase1/xgb_clr.model')

    ranker_predict(xgb_ranker, dtest, group_test)
    clr_predict(xgb_clr, dtest, threshold=0.5)


def main():
    build_ranker()
    build_classifier()


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")

    if True:
        sys.argv += ['--data_dir=../data/dedup/phase1/']
        FLAGS(sys.argv)
    else:
        app.run(main)
