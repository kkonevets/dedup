import deepmatcher as dm
import torch

the_dir = "./"

def tokenizer(s):
  return s.split()

train, vali, test = dm.data.process(
    path=the_dir,
    cache='cacheddata.pth',
    train='train.csv',
    validation='vali.csv',
    test='test.csv',
    tokenize=tokenizer,
    embeddings_cache_path=the_dir,
    embeddings='fasttext.ru.bin',
    pca=False)

model = dm.MatchingModel(attr_summarizer='hybrid')
best_save_path = the_dir+'hybrid_model.pth'

# model.load_state(best_save_path)

model.run_train(
    train,
    vali,
    epochs=6,
    batch_size=128,
    best_save_path=best_save_path,
    device=torch.device('cuda')
    # pos_neg_ratio=3
    )

# model.save_state(the_dir+'hybrid_model_10.pth')

########################### testing #####################################

model.load_state(best_save_path)
model.run_eval(test)

predictions = model.run_prediction(test)
predictions.head()

predictions.to_csv(the_dir+'preds.csv')

import tools
import pandas as pd
import scoring
import numpy as np

preds = pd.read_csv('../data/dedup/deepmatch/preds.csv')
preds.columns = ['id', 'prob']
 
preds.set_index('id', inplace=True)
samples = tools.load_samples('../data/dedup/samples.npz')
samples = samples[samples['ix']!=-1]
test = samples[samples['train']==0]
test = test.merge(preds, left_index=True, right_index=True)

recall_scale = scoring.get_recall_test_scale()
scoring.plot_precision_recall(test['target'], test['prob'], tag='_deep',
                                recall_scale=recall_scale)
scoring.topn_precision_recall_curve(test, 
                topns=[1,2,3,5,10,20], n_thresholds=100, tag='_deep')

texts = pd.read_csv('../data/dedup/deepmatch/test.csv')
texts = texts.merge(preds, on='id')
texts.set_index('id', inplace=True)
 
texts = texts.merge(test[['qid', 'synid', 'fid']], left_index=True, right_index=True)
texts = texts[['qid', 'synid', 'fid', 'label', 'prob', 'left_name', 'right_name']]
texts.sort_values(['qid', 'synid', 'label', 'fid'], 
                  ascending=[True, True, False, True], inplace=True)
texts = texts.astype({'qid':int, 'fid':int, 'label':int}, errors='ignore')
texts.to_csv('../data/dedup/deepmatch/test_watch.csv', index=False)

t1 = texts[(texts['label']==0)&(texts['prob']>0.9)]
t2 = texts[(texts['label']==1)&(texts['prob']<0.1)]
t1.to_csv('../data/dedup/deepmatch/typeI.csv')
t2.to_csv('../data/dedup/deepmatch/typeII.csv')


prob = preds['prob']
prob[prob>0.9].describe()
 
y_true, probas_pred = test['target'], test['prob']
cond = test['prob']>0.99
sub=test[cond][['target', 'prob']]
sub['target'].sum()/len(sub['prob'])