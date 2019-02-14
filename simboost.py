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
from preprocessing.letor import INFO_COLUMNS
import sys
from pymongo import MongoClient
from preprocessing.sampling import plot_topn_curves

FLAGS = flags.FLAGS
tools.del_all_flags(FLAGS)

flags.DEFINE_string("data_dir", None, "path to data directory")

matplotlib.use('agg')


def plot_precision_recall(y_true, probas_pred, tag='', recall_scale=1):
    import matplotlib.pyplot as plt

    average_precision = average_precision_score(y_true, probas_pred)
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)

    plt.clf()
    fig, ax = plt.subplots()

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'})
    ax.step(recall*recall_scale, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall*recall_scale, precision, alpha=0.2, color='b',
                    rasterized=True, **step_kwargs)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1])
    ax.set_xlim([0.0, 1])
    ax.set_title('Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    fig.savefig(FLAGS.data_dir + '/prec_recal%s.pdf' % tag, dpi=400)


def get_recall_scale():
    samples = tools.load_samples('../data/dedup/samples.npz')
    ptest = samples[(samples['target'] == 1) & (samples['train'] == 0)]
    counts = ptest['ix'].value_counts()
    recall_scale = 1-counts[-1]/counts.sum()
    return recall_scale


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


def clr_predict(probs, dmtx, threshold=0.4):
    y = dmtx.get_label()
    c = Counter(y)
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

    recall_scale = get_recall_scale()

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

    itest = pd.read_csv('../data/dedup/test_letor.ix',
                        header=None, sep='\t')
    itest.columns = ['qid', 'synid', 'fid', 'target']

    client = MongoClient(tools.c_HOST)
    db = client['cache']
    test_qids = itest['qid'].unique().tolist()
    positions_solr = db['solr_positions'].find(
        {'et_id': {'$in': test_qids}, 'i': {'$lte': max(group_test)-1}}, projection=['i'])
    positions_solr = pd.Series([p['i'] for p in positions_solr if p['i'] >= 0])

    plot_topn_curves([positions, positions_solr],
                     '../data/dedup/cumsum_test.pdf', scale=recall_scale,
                     labels=['reranking', 'SOLR'], title='Test: found in top N')


def build_classifier():
    ftrain = tools.load_samples('../data/dedup/train_sim_ftrs.npz')
    ftest = tools.load_samples('../data/dedup/test_sim_ftrs.npz')
    ftrain = ftrain[ftrain['ix'] != -1]
    ftest = ftest[ftest['ix'] != -1]

    recall_scale = get_recall_scale()

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
    _ = clr_predict(train_probs, dtrain)

    test_probs = xgb_clr.predict(dtest)
    y_pred = clr_predict(test_probs, dtest, threshold=0.5)
    plot_precision_recall(dtest.get_label(), test_probs, tag='20x20',
                          recall_scale=recall_scale)
    cm = confusion_matrix(dtest.get_label(), y_pred)
    print(cm)

    joblib.dump(xgb_clr, FLAGS.data_dir + '/xgb_clr.model')
    xgb_clr = joblib.load('../data/dedup/xgb_clr.model')

    # analyze
    itest = pd.read_csv('../data/dedup/test_letor.ix',
                        header=None, sep='\t')
    itest.columns = ['qid', 'synid', 'fid', 'target']
    itest['prob'] = xgb_clr.predict(dtest)
    itest['pred'] = (itest['prob'] > 0.8).astype(int)

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
