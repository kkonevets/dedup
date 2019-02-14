r"""
Sample command lines:

python3 simboost.py \
--data_dir=../data/dedup/ \

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
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
import matplotlib
from sklearn.utils.fixes import signature
from sklearn.externals import joblib
import sys
from pymongo import MongoClient
from preprocessing.sampling import plot_topn_curves

FLAGS = flags.FLAGS
tools.del_all_flags(FLAGS)

flags.DEFINE_string("data_dir", None, "path to data directory")

matplotlib.use('agg')


def plot_precision_recall(y_true, probas_pred, tag=''):
    import matplotlib.pyplot as plt

    average_precision = average_precision_score(y_true, probas_pred)
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)

    plt.clf()
    fig, ax = plt.subplots()

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'})
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, alpha=0.2, color='b',
                    rasterized=True, **step_kwargs)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1])
    ax.set_xlim([0.0, 1])
    ax.set_title('Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    fig.savefig(FLAGS.data_dir + '/prec_recal%s.pdf' % tag, dpi=400)


def ranker_predict(ranks, dmatrix, groups):
    y = dmatrix.get_label()
    positions = []
    ix_prev = 0
    scores = {i: [] for i in [1, 2, 5, 6, 7, 10]}
    for gcount in tools.tqdm(groups):
        y_cur = y[ix_prev: ix_prev + gcount]
        r = ranks[ix_prev: ix_prev + gcount]
        rsorted = y_cur[np.argsort(r)[::-1]]
        ix = np.where(rsorted == 1)[0][0]
        positions.append(ix)
        for k in scores.keys():
            val = tools.ndcg_at_k(rsorted, k, method=1)
            scores[k].append(val)
        ix_prev += gcount

    for k in list(scores.keys()):
        scores['ndcg@%d' % k] = np.round(np.mean(scores.pop(k)), 4)

    positions = pd.Series(positions)
    return scores, positions


def clr_predict(model, dmtx, threshold=0.4, tag=''):
    y = dmtx.get_label()
    c = Counter(y)
    probs = model.predict(dmtx)
    plot_precision_recall(y, probs, tag)
    y_pred = (probs >= threshold).astype(int)
    rep = classification_report(y, y_pred, labels=[1], output_dict=True)
    rep = rep['1']
    rep['base_accuracy'] = c[0]/sum(c.values())
    rep['accuracy'] = accuracy_score(y, y_pred)
    rep = {k: round(v, 4) for k, v in rep.items()}
    tools.pprint(rep)
    print('\n')
    return y_pred


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

    params = {
        'objective': 'rank:ndcg',
        'max_depth': 10,
        'eta': 0.1,
        'gamma': 1.0,
        'min_child_weight': 0.1,
        'eval_metric': ['ndcg@1', 'ndcg@2', 'map@2']
    }
    xgb_ranker = xgb.train(params, dtrain,
                           num_boost_round=5000,
                           early_stopping_rounds=20,
                           evals=[(dvali, 'vali')])

    ranks = xgb_ranker.predict(dtest)
    scores, positions = ranker_predict(ranks, dtest, group_test)
    tools.pprint(scores)
    # print(xgb_ranker.eval(dvali))

    joblib.dump(xgb_ranker, FLAGS.data_dir + '/xgb_ranker.model')
    xgb_ranker = joblib.load('../data/dedup/xgb_ranker.model')

    ixs_test = pd.read_csv('../data/dedup/test_letor.ix',
                           header=None, sep='\t')
    ixs_test.columns = ['qid', 'synid', 'fid', 'target']

    client = MongoClient(tools.c_HOST)
    db = client['cache']
    test_qids = ixs_test['qid'].unique().tolist()
    positions_solr = db['solr_positions'].find(
        {'et_id': {'$in': test_qids}, 'i': {'$lte': max(group_test)-1}}, projection=['i'])
    positions_solr = pd.Series([p['i'] for p in positions_solr])
    solr_top_total = len(positions_solr)
    positions_solr = positions_solr[positions_solr >= 0]
    scale = len(positions_solr)/solr_top_total

    plot_topn_curves([positions, positions_solr],
                     '../data/dedup/cumsum_test.pdf', scale=scale,
                     labels=['reranking', 'SOLR'], title='Test: found in top N')


def build_classifier():
    X_train, y_train = load_svmlight_file(FLAGS.data_dir + 'train_letor.txt')
    X_vali, y_vali = load_svmlight_file(FLAGS.data_dir + 'vali_letor.txt')
    X_test, y_test = load_svmlight_file(FLAGS.data_dir + 'test_letor.txt')

    notfound = tools.load_samples('../data/dedup/notfound.npz')

    size = len(X_test)/(len(X_train)+len(X_vali)+len(X_test))
    X_nf_part, X_nf_test = train_test_split(
        notfound, test_size=size, random_state=42)

    size = len(X_vali)/(len(X_train)+len(X_vali))
    X_nf_train, X_nf_vali = train_test_split(
        X_nf_part, test_size=size, random_state=42)

    X_train = np.vstack([X_train, X_nf_train])
    X_vali = np.vstack([X_vali, X_nf_vali])
    X_test = np.vstack([X_test, X_nf_test])

    y_train = np.hstack([y_train, np.zeros(X_nf_train.shape[0])])
    y_vali = np.hstack([y_vali, np.zeros(X_nf_vali.shape[0])])
    y_test = np.hstack([y_test, np.zeros(X_nf_test.shape[0])])

    dtrain = DMatrix(X_train, label=y_train)
    dvali = DMatrix(X_vali, label=y_vali)
    dtest = DMatrix(X_test, label=y_test)

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

    _ = clr_predict(xgb_clr, dtrain)
    y_pred = clr_predict(xgb_clr, dtest, threshold=0.8, tag='10_g10_m01')
    cm = confusion_matrix(dtest.get_label(), y_pred)
    print(cm)

    joblib.dump(xgb_clr, FLAGS.data_dir + '/xgb_clr.model')
    xgb_clr = joblib.load('../data/dedup/xgb_clr.model')

    ixs = pd.read_csv('../data/dedup/test_letor.ix', header=None, sep='\t')
    ixs.columns = ['qid', 'synid', 'fid', 'target']
    ixs['prob'] = xgb_clr.predict(dtest)
    ixs['pred'] = (ixs['prob'] > 0.8).astype(int)

    1

    # for max_depth in [3, 5, 8, 10, 15]:
    #     params['max_depth'] = max_depth
    #     xgb_clr = xgb.train(params, dtrain,
    #                         num_boost_round=1000,
    #                         early_stopping_rounds=10,
    #                         evals=[(dvali, 'vali')])
    #     y_pred = clr_predict(xgb_clr, dtest, threshold=0.5, tag=str(max_depth))


def test():
    dtest = xgb.DMatrix('../data/dedup/test_letor.txt')
    group_test = get_groups('../data/dedup/test_letor.group')

    xgb_ranker = joblib.load('../data/dedup/xgb_ranker.model')
    xgb_clr = joblib.load('../data/dedup/xgb_clr.model')

    ranker_predict(xgb_ranker, dtest, group_test)
    clr_predict(xgb_clr, dtest, threshold=0.5)


def main(argv):
    # build_ranker()
    build_classifier()


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")

    if True:
        sys.argv += ['--data_dir=../data/dedup/']
        FLAGS(sys.argv)
    else:
        app.run(main)
