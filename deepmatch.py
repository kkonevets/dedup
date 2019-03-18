import deepmatcher as dm

train, vali, test = dm.data.process(
    path='../data/dedup/deepmatch',
    cache='cacheddata.pth',
    train='train.csv',
    validation='vali.csv',
    embeddings='wiki.ru.bin',
    test='test.csv')

model.run_train(
    train,
    vali,
    epochs=10,
    batch_size=128,
    best_save_path='../data/dedup/deepmatch/hybrid_model.pth',
    pos_neg_ratio=3)