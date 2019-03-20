
import deepmatcher as dm

def tokenize(s):
    return s.split()

train, vali, test = dm.data.process(
    path='../data/dedup/deepmatch',
    cache='cacheddata.pth',
    train='train.csv',
    validation='vali.csv',
    test='test.csv',
    tokenize=tokenize,
    embeddings='fasttext.ru.bin',
    pca=False)

model = dm.MatchingModel(attr_summarizer='hybrid')

best_save_path='../data/dedup/deepmatch/hybrid_model.pth'

model.run_train(
    train,
    vali,
    epochs=10,
    batch_size=128,
    best_save_path=best_save_path,
    # pos_neg_ratio=3
    )

########################### testing #####################################

model.load_state(best_save_path)
model.run_eval(test)
