import tools
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymongo
from pymongo import MongoClient
import io
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def get_corpus(ets, total):
    for et in tqdm(ets, total=total):
        yield et


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

    client = MongoClient()
    dbm = client['release']

    upm = None
    samples = tools.load_samples('../data/dedup/samples.npz')

    translit = False
    if translit:
        corpus_file = '../data/dedup/corpus.npz'
    else:
        corpus_file = '../data/dedup/corpus_raw.npz'
        # .replace('й', 'и') - не делал так так как вектора странные

    def make_corpus():
        corpus = []

        subdf = samples[samples['synid'] == -1]
        subdf = subdf[['qid', 'train']].drop_duplicates()
        for _id, train in tqdm(subdf.values):
            et = fup.id2et[_id]
            text = tools.constitute_text(et['name'], et, fup)
            corpus.append((_id, None, None, train,
                           tools.normalize(text, translit=translit)))

        subdf = samples[samples['synid'] != -1]
        subdf = subdf[['synid', 'train']].drop_duplicates()
        for _id, train in tqdm(subdf.values):
            name, et = sid2et[_id]
            text = tools.constitute_text(name, et, fup)
            corpus.append((None, _id, None, train,
                           tools.normalize(text, translit=translit)))

        for et in tqdm(upm.ets):
            text = tools.constitute_text(et['name'], et, upm)
            corpus.append((None, None, et['id'], None,
                           tools.normalize(text, translit=translit)))

        corpus = np.array(corpus)
        columns = ['qid', 'synid', 'fid', 'train', 'text']
        np.savez(corpus_file, samples=corpus, columns=columns)

    make_corpus()
    corpus = tools.load_samples(corpus_file)
    corpus = corpus[corpus['train'] != 0]
    corpus = corpus['text'].values

    vectorizer = TfidfVectorizer(
        stop_words=get_stop_words(), token_pattern=r"(?u)\S+")
    model = vectorizer.fit(corpus)
    tools.do_pickle(model, '../data/dedup/tfidf_model.pkl')

    # sent = 'молоко пастеризованное домик в деревне'
    # model.transform([normalizer(sent)])


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
