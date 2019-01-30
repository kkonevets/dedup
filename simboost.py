import xgboost as xgb
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

#########################################################################


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
          'max_depth': 6,  # 10 best
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


def predict(model, X, y, threshold=0.35):
    c = Counter(y)
    probs = model.predict_proba(X)
    y_pred = (probs[:, 1] >= threshold).astype(int)
    print(classification_report(y, y_pred, labels=[1]))
    print('base accuracy %f' % (c[0]/sum(c.values())))
    print('accuracy %f' % accuracy_score(y, y_pred))
    return y_pred


_ = predict(model, X_train, y_train)
y_pred = predict(model, X_test, y_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
