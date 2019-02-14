r"""
Sample command lines:

python3 preprocessing/sampling.py \
--data_dir=../data/dedup \
--nrows=20 \
--nchoices=20 


"""
from absl import flags
from absl import app
from sklearn.model_selection import train_test_split
import string
import pymongo
from pymongo import MongoClient
from collections import Counter
from functools import partial
import re
import os
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

FLAGS = flags.FLAGS
# tools.del_all_flags(FLAGS)


flags.DEFINE_string("data_dir", None, "data directory path")
flags.DEFINE_integer("nrows", 100, "The TOP number of rows to query")
flags.DEFINE_integer("nchoices", 5, "The number of rows to sample from nrows")
flags.DEFINE_bool("for_test", False, "sample just for test")
flags.DEFINE_bool("no_prior", False, "sample just for test")
flags.DEFINE_string("mongo_host", tools.c_HOST, "MongoDb host")
flags.DEFINE_string("solr_host", tools.ml_HOST, "SOLR host")
flags.DEFINE_string("feed_db", '1cfreshv4', "feed mongodb database name")
flags.DEFINE_string("release_db", 'release', "master mongodb database name")

matplotlib.use('agg')

prog = re.compile("[\\W]", re.UNICODE)


def release_stat():
    client = MongoClient(FLAGS.mongo_host)
    db = client[FLAGS.release_db]
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
        'q=%s&rows=%d&fl=*,score' % (FLAGS.solr_host, quoted, rows)
    r = urllib.request.urlopen(q).read()
    docs = json.loads(r)['response']['docs']
    return docs


def plot_topn_curves(positions_list, fname, scale=1, labels=None, title=None):
    if labels is not None:
        assert len(positions_list) == len(labels)
    else:
        labels = [None]*len(positions_list)

    import matplotlib.pyplot as plt
    plt.clf()

    csums = []
    for positions in positions_list:
        rel1 = (positions.value_counts()/len(positions)).head(40)
        print(rel1)

        psex = positions[~positions.isin([-1, -2])]
        rel2 = (psex.value_counts()/len(positions))
        print(rel2.sum())
        print('\n')

        cumsum = rel2.sort_index().cumsum()
        cumsum.index += 1
        csums.append(cumsum)

    df = pd.DataFrame()
    for cumsum, label in zip(csums, labels):
        df[label] = cumsum

    df *= scale

    title = title if title else 'found in top N'
    ax = df.plot(title=title, grid=True)
    # xtics = list(df.index)[::2]
    # if xtics[-1] != df.index[-1]:
    #     xtics.append(df.index[-1])
    ax.set_xticks(df.index)
    fig = ax.get_figure()
    ax.set_xlabel("top N")
    ax.set_ylabel("recall")
    fig.savefig(fname)


def save_positions(positions):
    columns = ['et_id', 'et_name', 'et_brand',
               'el_id', 'el_name', 'el_brand', 'i']

    positions = [{k: pi for k, pi in zip(columns, p)} for p in positions]

    client = MongoClient(FLAGS.mongo_host)
    db = client['cache']
    db.drop_collection('solr_positions')
    db['solr_positions'].insert_many(positions)

    # positions.to_excel(FLAGS.data_dir + '/solr_positions.xlsx', index=False)

    positions = pd.Series([p['i'] for p in positions])
    plot_topn_curves([positions], FLAGS.data_dir + '/cumsum.pdf')


def get_prior(anew=True):
    prior_file = FLAGS.data_dir + '/priors.csv'
    if anew:
        client = MongoClient(FLAGS.mongo_host)
        db = client['cache']
        positions = db['solr_positions'].find({}, projection=['i'])
        positions = pd.Series([p['i'] for p in positions])
        prior = positions.value_counts()/positions.shape[0]
        prior.sort_index(ascending=True, inplace=True)
        prior.to_csv(prior_file, header=False)
    else:
        assert os.path.isfile(prior_file)
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
        rec += [et['srcId'], mname, '', -2]
        positions.append(rec)


def sample_one(found, et, synid, prior):
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

    if FLAGS.for_test or len(found) <= FLAGS.nchoices:
        ixs = df.index
    else:
        # prior index should be sorted in ascending order[-2,-1,0,1,2,3, ...]
        padded = np.pad(df['score'], (2, len(prior) - len(df) - 2),
                        mode='constant', constant_values=(df['score'].mean(), 0))
        padded /= padded.sum()
        # now -1 and -2 have mean probability, see TODO below
        p = np.multiply(padded, prior)
        p /= p.sum()
        last_ix = len(found) - 1
        ixs = np.random.choice([last_ix, last_ix] + prior.index[2:].tolist(), FLAGS.nchoices + 1,
                               replace=False, p=p)

        # TODO: sample randomly from out of range
        # out_of_range = set(ixs).intersection({-1, -2})
        # for ix in out_of_range:
        #     1

    # normalize scores
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


def query_one(id2brand, prior, et):
    client = MongoClient(FLAGS.mongo_host)
    mdb = client[FLAGS.release_db]

    et['id'] = et.pop('_id')
    bcs = set([int(c) for c in et['barcodes']])
    bid = et.get('brandId')
    bname = ''
    if bid:
        bname = id2brand[bid]['name']

    met = mdb.etalons.find_one(
        {'_id': et['srcId']}, projection=['name', 'synonyms'])
    msyns = ' '.join([s['name'] for s in met.get('synonyms', [])])
    msplited = set(prog.sub(' ', met['name'] + ' ' + msyns).lower().split())

    samples, positions = [], []

    for syn in et.get('synonyms', []):
        curname = syn['name'] + ' ' + bname
        splited = prog.sub(' ', curname).lower().split()

        common = msplited.intersection(splited)
        if common == set() or len(splited) <= 2:
            continue

        curname = tools.normalize(curname)
        if curname.strip() == '':
            continue
        found = query_solr(curname, FLAGS.nrows)

        if not FLAGS.no_prior and len(found):
            samples += sample_one(found, et, syn['id'], prior)

        if not FLAGS.for_test:
            append_position(positions, found, et, curname,
                            met['name'], bname, bcs)

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


def solr_sample(existing):
    client = MongoClient(FLAGS.mongo_host)
    db = client[FLAGS.feed_db]

    id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}

    positions, samples = [], []
    np.random.seed(0)

    if FLAGS.for_test or FLAGS.no_prior:
        prior = None
    else:
        prior = get_prior(anew=True)

    wraper = partial(query_one, id2brand, prior)

    def do_iterate(db, existing):
        for el in existing:
            et = db.etalons.find_one({'_id': el['_id']})
            et['srcId'] = el['_id_release']
            yield et

    nworkers = mp.cpu_count()
    with mp.Pool(20) as p:  # maxtasksperchild=5000
        with tqdm(total=len(existing)) as pbar:
            for samps, poss in tqdm(p.imap_unordered(wraper, do_iterate(db, existing))):
                samples += samps
                positions += poss
                pbar.update()
                # if len(positions) > 1000:
                #     break

    if FLAGS.no_prior:
        save_positions(positions)
        return

    samples = pd.DataFrame(samples)
    columns = ['qid', 'synid', 'fid', 'score', 'target', 'ix']  # , 'train'
    samples.columns = columns

    # synids_exclude = set(samples[samples['ix'] == -1]['synid'].unique())
    # cond = samples['synid'].isin(synids_exclude)

    # # saeve not found queries for separate testing
    # notfound = samples[cond].values.astype(np.float32)
    # np.savez(FLAGS.data_dir + '/notfound.npz',
    #          samples=notfound, columns=samples.columns)

    # # can't train models without target label
    # samples = samples[~cond]

    if not FLAGS.for_test:
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
        samples = tools.load_samples('../data/dedup/samples.npz')
        qids = set(samples[samples['train'] == 0]['qid'].unique())
        filtered = [el for el in existing if el['_id'] in qids]
        solr_sample(filtered)
    else:
        solr_sample(existing)


if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")

    if False:
        sys.argv += ['--data_dir=../data/dedup/']
        FLAGS(sys.argv)
    else:
        app.run(main)
