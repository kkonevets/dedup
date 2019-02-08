r"""
Sample command lines:

python3 preprocessing/map.py \
--samples_file=../data/dedup/samples_test.npz \
--corpus_file=../data/dedup/corpus_test.npz 

"""

from absl import flags
from absl import app
import tools
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymongo
from pymongo import MongoClient
import io
import sys
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

flags.DEFINE_bool("notranslit", False,
                  "don't transliterate english to cyrillic")
flags.DEFINE_bool("build_tfidf", False, "build tfidf model")
flags.DEFINE_string("samples_file", None, "path to samples numpy file")
flags.DEFINE_string("corpus_file", None, "path to save corpus numpy file")
flags.DEFINE_string("mongo_host", tools.c_HOST, "MongoDb host")
flags.DEFINE_string("feed_db", '1cfreshv4', "feed mongodb database name")
flags.DEFINE_string("release_db", 'release', "master mongodb database name")

FLAGS = flags.FLAGS


def get_stop_words():
    words = set(stopwords.words("russian"))
    with io.open('./solr/stopwords_ru.txt', encoding='utf8') as f:
        for l in f:
            splited = l.split('|')
            word = splited[0].strip()
            if word:
                words.update([word])

    return words


def make_corpus():
    client = MongoClient(FLAGS.mongo_host)
    db = client[FLAGS.feed_db]
    mdb = client[FLAGS.release_db]

    corpus = []
    id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}
    mid2brand = {c['_id']: c for c in mdb.brands.find({}, projection=['name'])}

    sid2et = {s['id']: (s['name'], e) for e in db.etalons.find({})
              for s in e.get('synonyms', [])}

    samples = tools.load_samples(FLAGS.samples_file)
    if 'train' not in samples.columns:
        samples['train'] = 0  # test

    translit = not FLAGS.notranslit

    subdf = samples[samples['synid'] == -1]
    subdf = subdf[['qid', 'train']].drop_duplicates()
    for _id, train in tqdm(subdf.values):
        et = db.etalons.find_one({'_id': _id})
        text = tools.constitute_text(et['name'], et, id2brand)
        corpus.append((_id, None, None, train,
                       tools.normalize(text, translit=translit)))

    subdf = samples[samples['synid'] != -1]
    subdf = subdf[['synid', 'train']].drop_duplicates()
    for _id, train in tqdm(subdf.values):
        name, et = sid2et[_id]
        text = tools.constitute_text(name, et, id2brand)
        corpus.append((None, _id, None, train,
                       tools.normalize(text, translit=translit)))

    if FLAGS.build_tfidf:
        ids = [et['_id'] for et in db.etalons.find({}, projection=['_id'])]
    else:
        ids = set(samples['fid'].unique())

    for _id in tqdm(ids):
        et = mdb.etalons.find_one({'_id': _id})
        text = tools.constitute_text(et['name'], et, mid2brand)
        corpus.append((None, None, et['_id'], None,
                       tools.normalize(text, translit=translit)))

    corpus = np.array(corpus)
    columns = ['qid', 'synid', 'fid', 'train', 'text']
    np.savez(FLAGS.corpus_file, samples=corpus, columns=columns)

    return corpus, columns


def get_tfidf(corpus, columns):
    '''
    Build tf-idf model using master data and 1c-Fresh train data
    '''
    corpus = pd.DataFrame(corpus)
    corpus.columns = columns
    corpus = corpus[corpus['train'] != 0]
    if len(corpus) == 0:  # test
        return
    corpus = corpus['text'].values

    vectorizer = TfidfVectorizer(
        stop_words=get_stop_words(), token_pattern=r"(?u)\S+")
    model = vectorizer.fit(corpus)
    tools.do_pickle(model, '../data/dedup/tfidf_model.pkl')

    # sent = 'молоко пастеризованное домик в деревне'
    # model.transform([normalizer(sent)])


def main(argv):
    del argv  # Unused.
    corpus, columns = make_corpus()
    if FLAGS.build_tfidf:
        get_tfidf(corpus, columns)


if __name__ == '__main__':
    flags.mark_flag_as_required("samples_file")
    flags.mark_flag_as_required("corpus_file")

    if False:
        sys.argv += ['--samples_file=../data/dedup/samples_test.npz',
                     '--corpus_file=../data/dedup/corpus_test.npz']
        FLAGS(sys.argv)
    else:
        app.run(main)
