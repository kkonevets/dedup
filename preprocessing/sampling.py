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
import re
import os
import pandas as pd
import numpy as np
from urllib.parse import quote
from tqdm import tqdm
import json
import tools
import sys
import scoring
import traceback
import aiohttp
import asyncio
import itertools
from tokenizer import tokenize

FLAGS = tools.FLAGS
SAMPLE_COLUMNS = ['qid', 'synid', 'fid', 'score', 'target', 'ix']  # , 'train'


async def query_solr(session, text, rows=1):
    quoted = quote('name:(%s)^5 || synonyms:(%s)' % (text, text))
    q = 'http://%s:8983/solr/nom_core/select?' \
        'q=%s&rows=%d&fl=*,score' % (FLAGS.solr_host, quoted, rows)
    async with session.get(q) as response:
        res = await response.text()
    # print(traceback.format_exc())
    return res


def save_positions(positions):
    columns = ['et_id', 'synid', 'el_id',  'i']
    positions = [{k: pi for k, pi in zip(columns, p)} for p in positions]

    client = MongoClient(FLAGS.mongo_host)
    db = client['cache']
    db.drop_collection('solr_positions')
    db['solr_positions'].insert_many(positions)

    positions = pd.Series([p['i'] for p in positions])
    scoring.plot_topn_curves([positions], FLAGS.data_dir + '/cumsum.pdf')


def get_position_record(found, et):
    bcs = set([int(c) for c in et['barcodes']])
    for i, el in enumerate(found):
        curbcs = [int(c) for c in el.get('barcodes', [])]
        if len(bcs.intersection(curbcs)):
            rec = [int(el['id']), i]
            break
    else:
        if len(found) == 0:
            rec = [None, -1]
        else:
            rec = [et['srcId'], -2]

    return rec


def sample_one(found, et, synid):
    df = [[et['id'],
           synid,
           int(el['id']),
           el['score'],
           int(int(el['id']) == et['srcId']),
           i]
          for i, el in enumerate(found)]

    score_ix = SAMPLE_COLUMNS.index('score')
    target_ix = SAMPLE_COLUMNS.index('target')

    # normalize scores
    score_sum = sum((el[score_ix] for el in df))
    has_traget = False
    for row in df:
        row[score_ix] = row[score_ix]/score_sum
        has_traget = max(has_traget, row[target_ix] == 1)

    if FLAGS.for_test:
        return df

    if not has_traget:
        # target is not in the TOP N
        df[0] = [et['id'], synid, et['srcId'], 0, 1, -1]

    return df


async def produce(queue, session, bname, et):
    et['id'] = et.pop('_id')

    def gen_names():
        if FLAGS.for_test:
            yield et['name'], None
        else:
            for syn in et.get('synonyms', []):
                yield syn['name'], syn['id']

    for curname, sid in gen_names():
        curname += ' ' + bname
        curname = tokenize(curname)
        if curname.strip() == '':
            continue
        res = await query_solr(session, curname, FLAGS.nrows)
        await queue.put((et, sid, res))


async def consume(queue, samples, positions):
    i = 0
    while True:
        et, sid, res = await queue.get()
        found = json.loads(res)['response']['docs']
        if len(found):
            samples += sample_one(found, et, sid)

        if not FLAGS.for_test:
            rec = get_position_record(found, et)
            positions.append([et['id'], sid] + rec)

        queue.task_done()
        i += 1
        if i % 5000 == 0:
            print(i)


async def query_all(elements):
    client = MongoClient(FLAGS.mongo_host)
    db = client[FLAGS.feed_db]

    id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}
    id2el = {el['_id']: el for el in elements}
    ets = db.etalons.find({'_id': {'$in': list(id2el)}})

    queue = asyncio.Queue(maxsize=10)

    samples, positions = [], []
    consumer = asyncio.create_task(consume(queue, samples, positions))
    producers = []

    timeout = aiohttp.ClientTimeout(total=60*60)  # 60 mins in keep alive mode
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for et in ets:
            if FLAGS.for_test:
                et['srcId'] = None
            else:
                el = id2el[et['_id']]
                et['srcId'] = el['_id_release']

            bid = et.get('brandId')
            bname = id2brand[bid]['name'] if bid else ''
            task = asyncio.create_task(produce(queue, session, bname, et))
            producers.append(task)

        await asyncio.wait(producers)
        await queue.join()
        consumer.cancel()

    return samples, positions


def solr_sample(elements):
    samples, positions = asyncio.run(query_all(elements))

    samples = pd.DataFrame(samples)
    samples.columns = SAMPLE_COLUMNS  # + train

    if not FLAGS.for_test:
        if FLAGS.nrows == 100:
            save_positions(positions)
        qids_train, qids_test = train_test_split(
            samples['qid'].unique(), test_size=0.2, random_state=11)
        samples['train'] = samples['qid'].isin(qids_train).astype(int)

    X_samples = samples.values.astype(np.float32)
    np.savez(FLAGS.data_dir + '/samples.npz',
             samples=X_samples, columns=samples.columns)


def get_existing(anew=False):
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


def main(argv):
    del argv  # Unused.

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    existing = get_existing(anew=False)
    # elements = existing

    if FLAGS.for_test:
        client = MongoClient(FLAGS.mongo_host)
        db = client[FLAGS.feed_db]
        not_existing = db.etalons.find(
            {'_id': {'$nin': [el['_id'] for el in existing]}},
            projection=['_id'])
        not_existing = [el for el in not_existing][:50000]
        solr_sample(not_existing)
    else:
        solr_sample(existing)


if __name__ == '__main__':
    import __main__
    flags.mark_flag_as_required("data_dir")

    if hasattr(__main__, '__file__'):
        app.run(main)
    else:
        sys.argv += ['--data_dir=../data/dedup/']
        FLAGS(sys.argv)
