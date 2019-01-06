from sklearn.feature_extraction.text import CountVectorizer
import tools
import pandas as pd
from tqdm import tqdm
import pymongo
from pymongo import MongoClient


def get_corpus(ets, total):
    for et in tqdm(ets, total=total):
        yield et


def main():
    client = MongoClient()
    db = client['master']

    brands = db.brands.find({}, projection=['_id', 'name'],
                            no_cursor_timeout=True)

    brands = list(get_corpus(brands, db.brands.count_documents({})))

    ets = db.etalons.find({}, projection=['_id', 'name', 'brandId', 'synonyms'],
                          no_cursor_timeout=True)
    ets = list(get_corpus(ets, db.etalons.count_documents({})))

    tools.do_pickle([ets, brands], '../data/dedup/master_ets_brands.pkl')

    ets, brands = tools.do_unpickle('../data/dedup/master_ets_brands.pkl')

    vec = CountVectorizer().fit((e['name'] for e in ets))
    bag_of_words = vec.transform((e['name'] for e in ets))
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    df = pd.DataFrame.from_records(words_freq)
    df.to_excel('../data/dedup/words_freq.xlsx', index=False)

    [el for el in words_freq if el[0] == 'миллиграмм']


if __name__ == "__main__":
    main()
