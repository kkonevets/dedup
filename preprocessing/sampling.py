import tools
import json
import urllib
import string
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from urllib.parse import quote
import pickle
import matplotlib
import numpy as np
import pandas as pd
import re
from functools import partial
from collections import Counter
from pymongo import MongoClient
import string
from sklearn.model_selection import train_test_split

matplotlib.use('agg')

prog = re.compile("[\\W]", re.UNICODE)


def master_stat():
    client = MongoClient()
    db = client['master']
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
    quoted = quote('name:(%s)^2 || synonyms:(%s)' % (text, text))
    q = 'http://c:8983/solr/nom_core/select?' \
        'q=%s&rows=%d&fl=*,score' % (quoted, rows)
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

    return prior


def append_position(positions, found, et, curname, mname, bname, bcs):
    rec = [et['id'], curname, bname]
    if len(found) == 0:
        rec += [None, mname, '', -1]
        positions.append(rec)

    for i, el in enumerate(found):
        curbcs = [int(c) for c in el.get('barcodes', [])]
        if len(bcs.intersection(curbcs)):
            rec += [int(el['id']), el['name'], el.get('brand', ''), i]
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


def query_one(id2brand, nrows, nchoices, prior, positions, samples, et):
    et['id'] = et.pop('_id')
    if not '#single' in et['comment']:
        return
    bcs = set([int(c) for c in et['barcodes']])
    bid = et.get('brandId')
    bname = ''
    if bid:
        bname = id2brand[bid]['name']

    client = MongoClient()
    mdb = client['master']
    met = mdb.etalons.find_one(
        {'_id': et['srcId']}, projection=['name', 'synonyms'])
    msyns = ' '.join([s['name'] for s in met.get('synonyms', [])])
    msplited = prog.sub(' ', met['name'] + ' ' + msyns).lower().split()

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


def solr_sample():
    client = MongoClient()
    db = client['1cfreshv4']
    total = db.etalons.count_documents({})
    id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}

    positions = []
    nrows = 100
    nchoices = 5
    np.random.seed(0)

    samples = []
    prior = get_prior(anew=False)
    assert prior.index.isin([-1, -2]).sum() == 2

    wraper = partial(query_one, id2brand, nrows,
                     nchoices, prior, positions, samples)

    iterator = db.etalons.find({})
    with mp.Pool(mp.cpu_count(), maxtasksperchild=100000) as p:
        with tqdm(total=total) as pbar:
            for _ in tqdm(p.imap_unordered(wraper, iterator)):
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


if __name__ == "__main__":
    pass
    # solr_sample()
