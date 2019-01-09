import tools
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymongo
from pymongo import MongoClient
import io
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def get_corpus(ets, total):
    for et in tqdm(ets, total=total):
        yield et


def dump_data():
    client = MongoClient()
    db = client['master']

    brands = db.brands.find({}, projection=['_id', 'name'],
                            no_cursor_timeout=True)
    brands = list(get_corpus(brands, db.brands.count_documents({})))

    ets = db.etalons.find({}, projection=['_id', 'name', 'brandId', 'synonyms'],
                          no_cursor_timeout=True)
    ets = list(get_corpus(ets, db.etalons.count_documents({})))

    tools.do_pickle([ets, brands], '../data/dedup/master_ets_brands.pkl')

    # ets, brands = tools.do_unpickle('../data/dedup/master_ets_brands.pkl')

    # vec = CountVectorizer().fit((e['name'] for e in ets))
    # bag_of_words = vec.transform((e['name'] for e in ets))
    # sum_words = bag_of_words.sum(axis=0)
    # words_freq = [(word, sum_words[0, idx])
    #               for word, idx in vec.vocabulary_.items()]
    # words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    # df = pd.DataFrame.from_records(words_freq)
    # df.to_excel('../data/dedup/words_freq.xlsx', index=False)

    # [el for el in words_freq if el[0] == 'миллиграмм']


def get_stop_words():
    words = set(stopwords.words("russian"))
    with io.open('./solr/stopwords_ru.txt', encoding='utf8') as f:
        for l in f:
            splited = l.split('|')
            word = splited[0].strip()
            if word:
                words.update([word])

    return words


def get_tfidf():
    '''
    Build tf-idf model using master data and 1c-Fresh train data
    '''
    with open('../data/1cfresh/1cfreshv4.json', 'r') as f:
        fresh = json.load(f)
        fup = tools.Updater(fresh)
        sid2et = {s['id']: (s['name'], e) for e in fup.ets
                  for s in e.get('synonyms', [])}

    upm = tools.load_master()
    npzfile = np.load('../data/dedup/train_samples.npz')
    train_samples = pd.DataFrame(npzfile['samples'])
    train_samples.columns = npzfile['columns']

    def corpus():
        sids = {_id for _id in train_samples['synid'].unique() if _id != -1}
        for _id in tqdm(sids):
            name, et = sid2et[_id]
            text = tools.constitute_text(name, et, fup)
            yield tools.normalize(text, True)

        subdf = train_samples[train_samples['synid'] == -1]
        for _id in tqdm(subdf['qid'].unique()):
            et = fup.id2et[_id]
            text = tools.constitute_text(et['name'], et, fup)
            yield tools.normalize(text, True)

        for et in tqdm(upm.ets):
            text = tools.constitute_text(et['name'], et, upm)
            yield tools.normalize(text, True)

    vectorizer = TfidfVectorizer(
        stop_words=get_stop_words(), token_pattern=r"(?u)\S+")
    model = vectorizer.fit(corpus())
    tools.do_pickle(model, '../data/dedup/tfidf_model.pkl')

    # sent = 'молоко пастеризованное домик в деревне'
    # model.transform([tools.normalize(sent, True)])


if __name__ == "__main__":
    pass

    # corpus = [
    #     'This is 10*20 first ?.. document.',
    #     'This document is the second document.',
    #     'And this is the third one.',
    #     'hy 4.5 коричневый махагон крем краска для волос серии professional coloring 100.0 мл',
    # ]
    # vectorizer = TfidfVectorizer(token_pattern=r"(?u)\S+")
    # X = vectorizer.fit_transform(corpus)
    # print(vectorizer.get_feature_names())

    1
