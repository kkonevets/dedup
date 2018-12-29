import json
import urllib
import string
from tqdm import tqdm
import pandas as pd
from urllib.parse import quote
import pymongo
from pymongo import MongoClient
import pickle

import utils


def query_solr(text, rows=1, exclude=[]):
    q = 'http://c:8983/solr/nom_core/select?df=my_text_ru' \
        '&q=%s&rows=%d' % (quote(text), rows)
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


def solr_stat():
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
    chars = string.punctuation
    chars = chars.replace('%', '').replace('_', '').replace('@', '')
    tt = str.maketrans(dict.fromkeys(chars, ' '))

    for et in tqdm(singles):
        bcs = set([int(c) for c in et['barcodes']])
        name = et['name'].translate(tt).lower()
        bid = et.get('brandId')
        bname = ''
        if bid:
            bname = id2brand[bid]['name']
            if bname.lower() not in name:
                bname = ''

        text = name + bname.translate(tt).lower()
        found = query_solr(text, nrows)

        met = mid2et[et['srcId']]
        mname = met['name'].translate(tt).lower()

        names = [name] + [s['name'].translate(tt).lower()
                          for s in et.get('synonyms', [])]

        for curname in names:
            rec = [et['id'], curname, bname]
            if len(found) == 0:
                rec += [None, '', '', -1]
                positions.append(rec)

            for i, el in enumerate(found):
                curbcs = [int(c) for c in el.get('barcodes', [])]
                if len(bcs.intersection(curbcs)):
                    rec += [int(el['id']), el['name'], el.get('brand', ''), i]
                    positions.append(rec)
                    break

            if len(rec) == 3:
                rec += [None, '', '', -2]
                positions.append(rec)

        # if len(positions) > 10:
        #     break

    positions = pd.DataFrame.from_records(positions)
    positions.columns = ['et_id', 'et_name', 'et_brand',
                         'el_id', 'el_name', 'el_brand', 'i']
    positions['equal'] = positions['et_name'].apply(
        lambda s: s.translate(tt).lower()) == positions['el_name'].apply(lambda s: str.lower(s))
    positions.to_excel('../data/dedup/solr_positions.xlsx', index=False)
    positions['i'].value_counts()/positions.shape[0]

    # ax = positions.plot(kind='hist', title='positions')
    # fig = ax.get_figure()
    # fig.savefig('../data/dedup/solr_positions.pdf')
