from absl import flags
from absl import app
from sklearn.model_selection import train_test_split
import string
import pymongo
from pymongo import MongoClient
from collections import Counter
from functools import partial
import re
import pandas as pd
import numpy as np
import matplotlib
import pickle
from urllib.parse import quote
import multiprocessing as mp
from tqdm import tqdm
import urllib
import json
import tools
import sys


flags.DEFINE_integer("train_batch_size", 32, "The batch size for training.")

FLAGS = flags.FLAGS

matplotlib.use('agg')

prog = re.compile("[\\W]", re.UNICODE)

MONGO_HOST = tools.c_HOST
SOLR_HOST = tools.ml_HOST


def release_stat():
    client = MongoClient(MONGO_HOST)
    db = client['release']
    total = db.etalons.count_documents({})

    iterator = db.etalons.find({}, projection=['id', 'name', 'synonyms'])
    ets = []
    nocommon = set()
    for et in tqdm(iterator, total=total):
        nsplited = prog.sub(' ', et['name']).lower().split()
        name = ' '.join(nsplited)
        syns = []
        for s in et.get('synonyms', []):
            splited = prog.sub(' ', s['name']).lower().split()
            sname = ' '.join(splited)
            common = set(nsplited).intersection(splited)
            if sname != name and len(splited) > 2:
                if len(common):
                    syns.append(s)
                else:
                    nocommon.update([et['_id']])
        if not syns:
            continue
        et['synonyms'] = syns
        ets.append(et)


def query_solr(text, rows=1):
    quoted = quote('name:(%s)^5 || synonyms:(%s)' % (text, text))
    q = 'http://%s:8983/solr/nom_core/select?' \
        'q=%s&rows=%d&fl=*,score' % (SOLR_HOST, quoted, rows)
    r = urllib.request.urlopen(q).read()
    docs = json.loads(r)['response']['docs']
    return docs


def save_positions(positions, nrows):
    positions = pd.DataFrame.from_records(positions)
    positions.columns = ['et_id', 'et_name', 'et_brand',
                         'el_id', 'el_name', 'el_brand', 'i']
    positions['equal'] = positions['et_name'].apply(
        tools.normalize) == positions['el_name'].apply(tools.normalize)
    positions.to_excel('../data/dedup/solr_positions.xlsx', index=False)

    rel1 = (positions['i'].value_counts()/positions.shape[0]).head(40)
    print(rel1)

    # positions = pd.read_excel('../data/dedup/solr_positions.xlsx')
    excl = positions[~positions['i'].isin([-1, -2])]
    rel2 = (excl['i'].value_counts()/positions.shape[0])
    print(rel2.sum())

    import matplotlib.pyplot as plt

    cumsum = rel2.sort_index().cumsum()
    plt.clf()
    ax = cumsum.plot(xlim=[0, nrows], ylim=[cumsum.min(), cumsum.max()],
                     title='SOLR found in top N', grid=True)
    fig = ax.get_figure()
    ax.set_xlabel("top N")
    ax.set_ylabel("recall")
    fig.savefig('../data/dedup/cumsum.pdf')

    incl = positions[positions['i'].isin([-1, -2])]
    incl['i'].value_counts()
    # pos_sort = positions.sort_values('i', ascending=False)


def get_prior(anew=False):
    prior_file = '../data/dedup/priors.csv'
    if anew:
        positions = pd.read_excel('../data/dedup/solr_positions.xlsx')
        prior = positions['i'].value_counts()/positions.shape[0]
        prior.sort_index(ascending=True, inplace=True)
        prior.to_csv(prior_file)
    else:
        prior = pd.read_csv(prior_file, header=None, index_col=0).loc[:, 1]

    assert prior.index.isin([-1, -2]).sum() == 2
    return prior


def append_position(positions, found, et, curname, mname, bname, bcs):
    rec = [et['id'], curname, bname]
    if len(found) == 0:
        rec += [None, mname, '', -1]
        positions.append(rec)

    for i, el in enumerate(found):
        curbcs = [int(c) for c in el.get('barcodes', [])]
        if len(bcs.intersection(curbcs)):
            rec += [int(el['id']), el.get('name', ''), el.get('brand', ''), i]
            positions.append(rec)
            break

    if len(rec) == 3:
        rec += [None, mname, '', -2]
        positions.append(rec)


def sample_one(found, et, nchoices, prior, synid):
    df = [(et['id'],
           synid,
           int(el['id']),
           el['score'],
           int(int(el['id']) == et['srcId']),
           None)
          for el in found]
    df = pd.DataFrame.from_records(df)
    df.columns = ['qid', 'synid', 'fid', 'score', 'target', 'ix']
    df['ix'] = df.index

    if len(found) > nchoices:
        # prior index should be sorted in ascending order[-2,-1,0,1,2,3, ...]
        padded = np.pad(df['score'], (2, len(prior) - len(df) - 2),
                        mode='constant', constant_values=(df['score'].mean(), 0))
        padded /= padded.sum()
        # now -1 and -2 have mean probability, see TODO below
        p = np.multiply(padded, prior)
        p /= p.sum()
        last_ix = len(found) - 1
        ixs = np.random.choice([last_ix, last_ix] + prior.index[2:].tolist(), nchoices + 1,
                               replace=False, p=p)

        # TODO: sample randomly from out of range
        # out_of_range = set(ixs).intersection({-1, -2})
        # for ix in out_of_range:
        #     1
    else:
        ixs = df.index

    df['score'] /= df['score'].sum()
    samples = df.loc[ixs]
    values = samples.values.tolist()
    if samples['target'].max() == 0:
        target = df[df['target'] == 1]
        if len(target):
            values[0] = target.iloc[0].values.tolist()
        else:
            # target is not in the TOP
            values[0] = [et['id'], synid, et['srcId'], 0, 1, -1]

    return values


def query_one(id2brand, nrows, nchoices, prior, et):
    et['id'] = et.pop('_id')
    bcs = set([int(c) for c in et['barcodes']])
    bid = et.get('brandId')
    bname = ''
    if bid:
        bname = id2brand[bid]['name']

    client = MongoClient(MONGO_HOST)
    mdb = client['release']
    met = mdb.etalons.find_one(
        {'_id': et['srcId']}, projection=['name', 'synonyms'])
    msyns = ' '.join([s['name'] for s in met.get('synonyms', [])])
    msplited = prog.sub(' ', met['name'] + ' ' + msyns).lower().split()

    samples, positions = [], []

    for syn in et.get('synonyms', []):
        curname = syn['name'] + ' ' + bname
        splited = prog.sub(' ', curname).lower().split()

        common = set(msplited).intersection(splited)
        if common == set() or len(splited) <= 2:
            continue

        curname = tools.normalize(curname)
        if curname.strip() == '':
            continue
        found = query_solr(curname, nrows)

        if len(found):
            samples += sample_one(found, et, nchoices, prior, syn['id'])

        append_position(positions, found, et, curname,
                        met['name'], bname, bcs)

    return samples, positions


def get_id2bc(dbname):
    client = MongoClient(MONGO_HOST)
    db = client[dbname]
    total = db.etalons.count_documents({})
    id2bc = []
    for et in tqdm(db.etalons.find({}), total=total):
        for bc in et.get('barcodes', []):
            id2bc.append((et['_id'], int(bc)))

    df = pd.DataFrame(id2bc)
    df.columns = ('_id', 'barcode')

    return df


def get_existing(anew=False):
    client = MongoClient(MONGO_HOST)
    db = client['cache']
    if anew:
        fresh_df = get_id2bc('1cfreshv4')
        fresh_df.columns = ('_id_fresh', 'barcode')
        release_df = get_id2bc('release')
        release_df.columns = ('_id_release', 'barcode')

        merged = fresh_df.merge(release_df, how='inner', on='barcode')
        merged = merged.groupby('_id_fresh').filter(lambda x: len(x) == 1)

        bulk = []
        for _id_fresh, barcode, _id_release in merged.values:
            bulk.append({'_id': int(_id_fresh),
                         '_id_release': int(_id_release),
                         'barcode': int(barcode)})

        db.drop_collection('1cfresh_existing')
        db['1cfresh_existing'].insert_many(bulk)
    else:
        bulk = [el for el in db['1cfresh_existing'].find({})]
        return bulk


def solr_sample():
    client = MongoClient(MONGO_HOST)
    db = client['1cfreshv4']

    existing = get_existing(anew=False)
    id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}

    positions, samples = [], []
    nrows = 100
    nchoices = 5
    np.random.seed(0)

    prior = get_prior(anew=True)

    wraper = partial(query_one, id2brand, nrows, nchoices, prior)

    def do_iterate():
        for el in existing:
            et = db.etalons.find_one({'_id': el['_id']})
            et['srcId'] = el['_id_release']
            yield et

    nworkers = mp.cpu_count()
    with mp.Pool(20) as p:  # maxtasksperchild=5000
        with tqdm(total=len(existing)) as pbar:
            for samps, poss in tqdm(p.imap_unordered(wraper, do_iterate())):
                samples += samps
                positions += poss
                pbar.update()
                # if len(positions) > 1000:
                #     break

    # npzfile = np.load('../data/dedup/samples.npz')
    # samples = npzfile['samples']
    samples = np.array(samples)
    columns = ['qid', 'synid', 'fid', 'score', 'target', 'ix']  # , 'train'

    samples = pd.DataFrame(samples)
    samples.columns = columns
    qids = samples['qid'].unique()
    qids_train, qids_test = train_test_split(
        qids, test_size=0.33, random_state=42)

    samples['train'] = samples['qid'].isin(qids_train).astype(int)

    np.savez('../data/dedup/samples.npz',
             samples=samples.values, columns=samples.columns)

    save_positions(positions, nrows)


def main(argv):
    del argv  # Unused.
    solr_sample()


if __name__ == '__main__':
    app.run(main)
