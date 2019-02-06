import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd
import shutil
import os
import tools
from preprocessing.textsim import get_sim_features
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def plain():
    def predict(model, X, y, threshold=0.4):
        c = Counter(y)
        probs = model.predict_proba(X)
        y_pred = (probs[:, 1] >= threshold).astype(int)
        print(classification_report(y, y_pred, labels=[1]))
        print('base accuracy %f' % (c[0]/sum(c.values())))
        print('accuracy %f' % accuracy_score(y, y_pred))
        return y_pred

    data_train = tools.load_samples('../data/dedup/train_sim_ftrs.npz', 'vals')
    data_test = tools.load_samples('../data/dedup/test_sim_ftrs.npz', 'vals')

    cols = [c for c in data_train.columns if c not in {
        'qid', 'synid', 'fid', 'target'}]

    norm = StandardScaler()
    X_train = norm.fit_transform(data_train[cols])
    X_test = norm.transform(data_test[cols])

    y_train = data_train['target']
    y_test = data_test['target']

    #########################################################################

    params = {'n_estimators': 500, 'n_jobs': -1,  # 1000 best
              'max_depth': 10,  # 10 best
              'learning_rate': 0.1,  # !!!!!!!!!!!!!!!
              #   'min_child_weight': 1,
              #   'gamma': 3,
              #   'subsample': 0.8,
              #   'colsample_bytree': 0.8,
              #   'reg_alpha': 5
              }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=True, early_stopping_rounds=10,
              eval_set=[(X_test, y_test)], eval_metric='logloss')

    _ = predict(model, X_train, y_train)
    y_pred = predict(model, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)


def ranking():
    x_train, y_train = load_svmlight_file("../data/dedup/train_letor.txt")
    x_valid, y_valid = load_svmlight_file("../data/dedup/vali_letor.txt")
    x_test, y_test = load_svmlight_file("../data/dedup/test_letor.txt")

    def predict(model, X, y, groups):
        ranks = model.predict(X)
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
            scores['ndcg@%d' % k] = np.mean(scores.pop(k))

        return scores

    def get_groups(fname):
        groups = []
        with open(fname, "r") as f:
            data = f.readlines()
            for line in data:
                groups.append(int(line.split("\n")[0]))
        return groups

    group_train = get_groups("../data/dedup/train_letor.group")
    group_valid = get_groups("../data/dedup/vali_letor.group")
    group_test = get_groups("../data/dedup/test_letor.group")

    train_dmatrix = DMatrix(x_train, y_train)
    valid_dmatrix = DMatrix(x_valid, y_valid)
    test_dmatrix = DMatrix(x_test)

    train_dmatrix.set_group(group_train)
    valid_dmatrix.set_group(group_valid)

    params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 10, 'eval_metric': ['ndcg@1', 'ndcg@2']}
    xgb_model = xgb.train(params, train_dmatrix,
                          num_boost_round=1000, early_stopping_rounds=20,
                          evals=[(valid_dmatrix, 'vali')])

    predict(xgb_model, valid_dmatrix, y_valid, group_valid)
