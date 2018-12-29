from sklearn.feature_extraction.text import CountVectorizer
import utils
import pandas as pd
from tqdm import tqdm
import pymongo
from pymongo import MongoClient


def get_corpus(ets, total):
    for et in tqdm(ets, total=total):
        sent = utils.normalize(et['name'])
        yield sent


def main():
    client = MongoClient()
    db = client['master']

    ets = db.etalons.find({}, projection=['name'],
                          no_cursor_timeout=True)

    # total = db.etalons.count_documents({})
    # corpus = list(get_corpus(ets, total))
    # utils.do_pickle(corpus, '../data/dedup/corpus.pkl')
    corpus = utils.do_unpickle('../data/dedup/corpus.pkl')

    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    df = pd.DataFrame.from_records(words_freq)
    df.to_excel('../data/dedup/words_freq.xlsx', index=False)

    [el for el in words_freq if el[0] == 'миллиграмм']


if __name__ == "__main__":
    pass
