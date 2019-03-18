
import deepmatcher as dm

train, vali, test = dm.data.process(
    path='../data/dedup/deepmatch',
    cache='cacheddata.pth',
    train='train.csv',
    validation='vali.csv',
    test='test.csv',
    tokenize=lambda s: s.split(),
    embeddings='fasttext.ru.bin',
    pca=False)

model = dm.MatchingModel(attr_summarizer='hybrid')

model.run_train(
    train,
    vali,
    epochs=10,
    batch_size=128,
    best_save_path='../data/dedup/deepmatch/hybrid_model.pth',
    # pos_neg_ratio=3
    )