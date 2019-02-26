r"""
Sample command lines:

python3 preprocessing/corpus.py \
--data_dir=../data/dedup \
--build_tfidf
"""

from absl import flags
from absl import app
import tools
import pandas as pd
import numpy as np
from tqdm import tqdm
from pymongo import MongoClient
import io
import sys
from nltk.corpus import stopwords
from tokenizer import tokenize


FLAGS = tools.FLAGS


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

    samples = tools.load_samples(FLAGS.data_dir + '/samples.npz')
    if 'train' not in samples.columns:
        samples['train'] = 0  # test

    # translit = not FLAGS.notranslit

    ###############################################################

    if FLAGS.build_tfidf:
        total = mdb.etalons.count_documents({})
        ets = mdb.etalons.find({}, projection=['name', 'brandId', 'synonyms'])
    else:
        ids = list(samples['fid'].unique())
        total = len(ids)
        ets = mdb.etalons.find({'_id': {'$in': ids}},
                               projection=['name', 'brandId', 'synonyms'])

    for et in tqdm(ets, total=total):
        text = tools.constitute_text(et['name'], et, mid2brand, use_syns=True)
        corpus.append((None, None, et['_id'], None,
                       tokenize(text)))

    ###############################################################

    subdf = samples[['qid', 'synid', 'train']].drop_duplicates()
    qids = subdf['qid'].unique().tolist()
    id2et = {et['_id']: et for et in db.etalons.find({'_id': {'$in': qids}})}
    for qid, sid, train in tqdm(subdf.values):
        et = id2et[qid]
        if pd.isna(sid):
            name = et['name']
        else:
            name = next((s['name'] for s in et.get('synonyms')
                         if s['id'] == sid))

        text = tools.constitute_text(name, et, id2brand, use_syns=False)
        corpus.append((qid, sid, None, train,
                       tokenize(text)))

    corpus = np.array(corpus)
    columns = ['qid', 'synid', 'fid', 'train', 'text']
    np.savez(FLAGS.data_dir + '/corpus.npz', samples=corpus, columns=columns)

    return corpus, columns


def get_tfidf(corpus, columns):
    '''
    Build tf-idf model using master data and 1c-Fresh train data
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer

    corpus = pd.DataFrame(corpus)
    corpus.columns = columns
    corpus = corpus[corpus['train'] != 0]
    if len(corpus) == 0:  # test
        return
    texts = corpus['text'].values

    vectorizer = TfidfVectorizer(
        stop_words=get_stop_words(), token_pattern=r"(?u)\S+")
    model = vectorizer.fit(texts)
    tools.do_pickle(model, '../data/dedup/tfidf_model.pkl')

    # sent = 'молоко пастеризованное домик в деревне'
    # model.transform([normalizer(sent)])


def main(argv):
    del argv  # Unused.
    corpus, columns = make_corpus()
    if FLAGS.build_tfidf:
        get_tfidf(corpus, columns)


if __name__ == '__main__':
    import __main__
    flags.mark_flag_as_required("data_dir")

    if hasattr(__main__, '__file__'):
        app.run(main)
    else:
        sys.argv += ['--data_dir=../data/dedup', ]
        FLAGS(sys.argv)
