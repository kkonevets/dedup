import utils
import json
import urllib
import string
from tqdm import tqdm
import pandas as pd
from urllib.parse import quote
import pickle
import matplotlib
import numpy as np
import pandas as pd
from collections import Counter

matplotlib.use('agg')


def query_solr(text, rows=1, exclude=[]):
    q = 'http://c:8983/solr/nom_core/select?df=my_text_ru' \
        '&q=%s&rows=%d&fl=*,score' % (quote(text), rows)
    r = urllib.request.urlopen(q).read()
    docs = json.loads(r)['response']['docs']
    return docs


def save_mid2et(singles):
    sids = {et['srcId'] for et in singles}
    master = utils.load_master()
    mdb = master['Database']
    id2et = {}
    for et in mdb['etalons']:
        if et['id'] in sids:
            id2et[et['id']] = et

    with open('../data/dedup/mid2et.pkl', 'wb') as f:
        pickle.dump(id2et, f)


def sample_one(found, et, nchoices, synid):
    np.random.seed(0)
    df = [(et['id'],
           synid,
           int(el['id']),
           el['score'],
           int(int(el['id']) == et['srcId']),
           -1)
          for el in found]
    df = pd.DataFrame.from_records(df)
    df.columns = ['qid', 'synid', 'fid', 'score', 'target', 'ix']
    df['score'] = df['score']/df['score'].sum()
    df['ix'] = df.index

    if len(found) > nchoices:
        ixs = np.random.choice(df.index, nchoices + 1,
                               replace=False, p=df['score'])
    else:
        ixs = df.index

    samples = df.loc[ixs]
    values = samples.values.tolist()
    if samples['target'].max() == 0:
        target = df[df['target'] == 1]
        if len(target):
            values[-1] = target.iloc[0].values.tolist()
        else:
            values[-1] = [et['id'], synid, et['srcId'], 0, 1, -1]

    return values


def solr_sample():
    with open('../data/1cfresh/1cfreshv4.json', 'r') as f:
        fresh = json.load(f)
        db = fresh['Database']
        singles = [et for et in db['etalons'] if '#single' in et['comment']]

    # save_mid2et(singles)
    with open('../data/dedup/mid2et.pkl', 'rb') as f:
        mid2et = pickle.load(f)

    id2brand = {b['id']: b for b in db['brands']}
    positions = []
    nrows = 100
    nchoices = 5

    samples = []

    for et in tqdm(singles):
        bcs = set([int(c) for c in et['barcodes']])
        bid = et.get('brandId')
        bname = ''
        if bid:
            bname = utils.normalize(id2brand[bid]['name'])

        name = utils.normalize(et['name'])

        met = mid2et[et['srcId']]
        mname = utils.normalize(met['name'])

        names = [(-1, name)] + [(s['id'], utils.normalize(s['name']))
                                for s in et.get('synonyms', [])]
        names = [n for n in names if n[1] != mname]

        for synid, curname in names:
            text = curname + ' ' + bname
            if text.strip() == '':
                continue
            found = query_solr(text, nrows)

            if len(found):
                samples += sample_one(found, et, nchoices, synid)

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

        if len(positions) > 1000:
            break

    samples = pd.DataFrame.from_records(samples, coerce_float=False)
    samples.columns = ['qid', 'synid', 'fid', 'score', 'target', 'ix']
    samples.to_excel('../data/dedup/samples.xlsx',
                     index=False, encoding='utf8')

    positions = pd.DataFrame.from_records(positions)
    positions.columns = ['et_id', 'et_name', 'et_brand',
                         'el_id', 'el_name', 'el_brand', 'i']
    positions['equal'] = positions['et_name'].apply(
        utils.normalize) == positions['el_name'].apply(utils.normalize)
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
    ax = cumsum.plot(xlim=[0, 100], ylim=[cumsum.min(), cumsum.max()],
                     title='SOLR found in top N', grid=True)
    fig = ax.get_figure()
    ax.set_xlabel("top N")
    ax.set_ylabel("recall")
    fig.savefig('../data/dedup/cumsum.pdf')

    incl = positions[positions['i'].isin([-1, -2])]
    incl['i'].value_counts()
    # pos_sort = positions.sort_values('i', ascending=False)


if __name__ == "__main__":
    pass
    # solr_sample()
