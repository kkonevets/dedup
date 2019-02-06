import xgboost as xgb
from xgboost import DMatrix
from sklearn.datasets import load_svmlight_file
import numpy as np
import pandas as pd
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


def do_rank():
    X_train, y_train = load_svmlight_file("../data/dedup/train_letor.txt")
    X_vali, y_vali = load_svmlight_file("../data/dedup/vali_letor.txt")
    X_test, y_test = load_svmlight_file("../data/dedup/test_letor.txt")

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
            scores['ndcg@%d' % k] = np.round(np.mean(scores.pop(k)), 4)

        return scores

    def get_groups(fname):
        groups = []
        with open(fname, "r") as f:
            data = f.readlines()
            for line in data:
                groups.append(int(line.split("\n")[0]))
        return groups

    group_train = get_groups("../data/dedup/train_letor.group")
    group_vali = get_groups("../data/dedup/vali_letor.group")
    group_test = get_groups("../data/dedup/test_letor.group")

    train_dmatrix = DMatrix(X_train, y_train)
    vali_dmatrix = DMatrix(X_vali, y_vali)
    test_dmatrix = DMatrix(X_test)

    train_dmatrix.set_group(group_train)
    vali_dmatrix.set_group(group_vali)

    eval_metric = ['ndcg@%d' % i for i in range(1, 3)] +\
        ['map@%d' % i for i in range(2, 3)]

    params = {'objective': 'rank:ndcg', 'eta': 0.1, 'gamma': 1.0,
              'min_child_weight': 0.1, 'max_depth': 10,
              'eval_metric': eval_metric}
    xgb_model = xgb.train(params, train_dmatrix,
                          num_boost_round=1000, early_stopping_rounds=20,
                          evals=[(vali_dmatrix, 'vali')])

    tools.pprint(predict(xgb_model, test_dmatrix, y_test, group_test))
    # print(xgb_model.eval(vali_dmatrix))

    ranks_train = xgb_model.predict(train_dmatrix)
    ranks_vali = xgb_model.predict(vali_dmatrix)
    ranks_test = xgb_model.predict(test_dmatrix)


def do_classify():
    def predict(model, X, y, threshold=0.4):
        c = Counter(y)
        probs = model.predict_proba(X)
        y_pred = (probs[:, 1] >= threshold).astype(int)
        print(classification_report(y, y_pred, labels=[1]))
        print('base accuracy %f' % (c[0]/sum(c.values())))
        print('accuracy %f' % accuracy_score(y, y_pred))
        return y_pred

    X_train, y_train = load_svmlight_file("../data/dedup/train_letor.txt")
    X_vali, y_vali = load_svmlight_file("../data/dedup/vali_letor.txt")
    X_test, y_test = load_svmlight_file("../data/dedup/test_letor.txt")

    params = {'n_estimators': 1000, 'n_jobs': -1,  # 1000 best
              'max_depth': 10,  # 10 best
              'learning_rate': 0.1,  # !!!!!!!!!!!!!!!
              #   'min_child_weight': 1,
              #   'gamma': 3,
              #   'subsample': 0.8,
              #   'colsample_bytree': 0.8,
              #   'reg_alpha': 5
              }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=True, early_stopping_rounds=20,
              eval_set=[(X_vali, y_vali)], eval_metric='logloss')

    _ = predict(model, X_train, y_train)
    y_pred = predict(model, X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
