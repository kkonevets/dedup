r"""
Sample command lines:

python3 preprocessing/sampling.py \
--data_dir=../data/dedup \
--nrows=2 \
--for_test \

"""

from absl import flags
from absl import app
from sklearn.model_selection import train_test_split
import pymongo
from pymongo import MongoClient
from functools import partial
import re
import os
import pandas as pd
import numpy as np
from urllib.parse import quote
import multiprocessing as mp
from tqdm import tqdm
import urllib
import json
import tools
import sys
import scoring
import traceback

FLAGS = tools.FLAGS
SAMPLE_COLUMNS = ['qid', 'synid', 'fid', 'score', 'target', 'ix']  # , 'train'


def query_solr(text, rows=1):
    quoted = quote('name:(%s)^5 || synonyms:(%s)' % (text, text))
    q = 'http://%s:8983/solr/nom_core/select?' \
        'q=%s&rows=%d&fl=*,score' % (FLAGS.solr_host, quoted, rows)
    r = urllib.request.urlopen(q).read()
    docs = json.loads(r)['response']['docs']
    # print(traceback.format_exc())
    return docs


def save_positions(positions):
    columns = ['et_id', 'et_name', 'et_brand',
               'el_id', 'el_name', 'el_brand', 'i']

    positions = [{k: pi for k, pi in zip(columns, p)} for p in positions]

    client = MongoClient(FLAGS.mongo_host)
    db = client['cache']
    db.drop_collection('solr_positions')
    db['solr_positions'].insert_many(positions)

    positions = pd.Series([p['i'] for p in positions])
    scoring.plot_topn_curves([positions], FLAGS.data_dir + '/cumsum.pdf')


def get_position_record(found, et, mname):
    bcs = set([int(c) for c in et['barcodes']])
    for i, el in enumerate(found):
        curbcs = [int(c) for c in el.get('barcodes', [])]
        if len(bcs.intersection(curbcs)):
            rec = [int(el['id']), el.get('name', ''), el.get('brand', ''), i]
            break
    else:
        if len(found) == 0:
            rec = [None, mname, '', -1]
        else:
            rec = [et['srcId'], mname, '', -2]

    return rec


def sample_one(found, et, synid):
    df = [(et['id'],
           synid,
           int(el['id']),
           el['score'],
           int(int(el['id']) == et['srcId']),
           None)
          for el in found]
    df = pd.DataFrame.from_records(df)
    df.columns = SAMPLE_COLUMNS
    df['ix'] = df.index

    # normalize scores
    df['score'] /= df['score'].sum()

    values = df.values.tolist()
    if FLAGS.for_test:
        return values

    if df['target'].max() == 0:
        # target is not in the TOP N
        values[0] = [et['id'], synid, et['srcId'], 0, 1, -1]

    return values


def query_one(id2brand, et):
    client = MongoClient(FLAGS.mongo_host)
    mdb = client[FLAGS.release_db]

    et['id'] = et.pop('_id')
    bid = et.get('brandId')
    bname = ''
    if bid:
        bname = id2brand[bid]['name']

    if not FLAGS.for_test:
        met = mdb.etalons.find_one(
            {'_id': et['srcId']}, projection=['name', 'synonyms'])
        msyns = ' '.join([s['name'] for s in met.get('synonyms', [])])
        metstr = met['name'] + ' ' + msyns
        msplited = set(tools.prog_with_digits.sub(' ', metstr).lower().split())
    else:
        msplited = set()

    samples, positions = [], []
    for syn in et.get('synonyms', []):
        curname = syn['name'] + ' ' + bname
        splited = tools.prog_with_digits.sub(' ', curname).lower().split()

        common = msplited.intersection(splited)
        if (not FLAGS.for_test and common == set()) or len(splited) <= 2:
            continue

        curname = tools.normalize(curname)
        if curname.strip() == '':
            continue
        found = query_solr(curname, FLAGS.nrows)

        if len(found):
            samples += sample_one(found, et, syn['id'])

        if not FLAGS.for_test:
            rec = get_position_record(found, et, met['name'])
            positions.append([et['id'], curname, bname] + rec)

    return samples, positions


def get_id2bc(dbname):
    client = MongoClient(FLAGS.mongo_host)
    db = client[dbname]
    total = db.etalons.count_documents({})
    id2bc = []
    for et in tqdm(db.etalons.find({}), total=total):
        for bc in et.get('barcodes', []):
            id2bc.append((et['_id'], int(bc)))

    df = pd.DataFrame(id2bc)
    return df


def get_existing(anew=False):
    client = MongoClient(FLAGS.mongo_host)
    db = client['cache']
    cache_name = '%s_existing' % FLAGS.feed_db
    if anew:
        feed_df = get_id2bc(FLAGS.feed_db)
        feed_df.columns = ('_id', 'barcode')
        release_df = get_id2bc(FLAGS.release_db)
        release_df.columns = ('_id_release', 'barcode')

        merged = feed_df.merge(release_df, how='inner', on='barcode')
        merged = merged.groupby('_id').filter(lambda x: len(x) == 1)

        bulk = []
        for _id, barcode, _id_release in merged.values:
            bulk.append({'_id': int(_id),
                         '_id_release': int(_id_release),
                         'barcode': int(barcode)})

        db.drop_collection(cache_name)
        db[cache_name].insert_many(bulk)
    else:
        bulk = [el for el in db[cache_name].find({})]
        return bulk


def solr_sample(elements):
    client = MongoClient(FLAGS.mongo_host)
    db = client[FLAGS.feed_db]

    id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}
    positions, samples = [], []

    wraper = partial(query_one, id2brand)

    def do_iterate(db, elements):
        for el in elements:
            et = db.etalons.find_one({'_id': el['_id']})
            if FLAGS.for_test:
                et['srcId'] = None
            else:
                et['srcId'] = el['_id_release']
            yield et

    # nworkers = mp.cpu_count()
    with mp.Pool(30) as p:  # maxtasksperchild=5000
        with tqdm(total=len(elements)) as pbar:
            # for samps, poss in tqdm(map(wraper, do_iterate(db, elements))):
            for samps, poss in tqdm(p.imap_unordered(wraper, do_iterate(db, elements))):
                samples += samps
                positions += poss
                pbar.update()

    samples = pd.DataFrame(samples)
    samples.columns = SAMPLE_COLUMNS  # + train

    if not FLAGS.for_test:
        # save_positions(positions)
        qids_train, qids_test = train_test_split(
            samples['qid'].unique(), test_size=0.2, random_state=11)
        samples['train'] = samples['qid'].isin(qids_train).astype(int)

    X_samples = samples.values.astype(np.float32)
    np.savez(FLAGS.data_dir + '/samples.npz',
             samples=X_samples, columns=samples.columns)


def main(argv):
    del argv  # Unused.

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    existing = get_existing(anew=False)

    if FLAGS.for_test:
        client = MongoClient(FLAGS.mongo_host)
        db = client[FLAGS.feed_db]
        not_existing = db.etalons.find(
            {'_id': {'$nin': [el['_id'] for el in existing]}},
            projection=['_id'])
        not_existing = [el for el in not_existing]
        solr_sample(not_existing)
    else:
        solr_sample(existing)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")

    if False:
        sys.argv += ['--data_dir=../data/dedup/']
        FLAGS(sys.argv)
    else:
        app.run(main)
