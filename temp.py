import io
import json
from tqdm import tqdm
import tools
import pandas as pd
from pymongo import MongoClient


with io.open('../data/release/1cnrel-2019-01-09-17-2019-02-04-09.json',
             encoding='utf8') as f:
    feed = json.load(f)

tools.feed2mongo(feed, 'release')


corpus = tools.load_samples(
    '../data/dedup/corpus.npz')

lens = []
for name in corpus['text'].values:
    splited = name.split()
    lens.append(len(splited))

lens = pd.DataFrame(lens)
for qa in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    print(qa, lens.quantile(qa)[0])
