import utils
import json
import urllib
import string
from tqdm import tqdm
import pandas as pd
from urllib.parse import quote
import pickle


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

    for et in tqdm(singles):
        bcs = set([int(c) for c in et['barcodes']])
        bid = et.get('brandId')
        bname = ''
        if bid:
            bname = utils.normalize(id2brand[bid]['name'])

        name = utils.normalize(et['name'])

        met = mid2et[et['srcId']]
        mname = utils.normalize(met['name'])

        names = [name] + [utils.normalize(s['name'])
                          for s in et.get('synonyms', [])]
        names = [n for n in names if n != mname]

        for curname in names:
            text = curname + ' ' + bname
            if text.strip() == '':
                continue
            found = query_solr(text, nrows)

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
        utils.normalize) == positions['el_name'].apply(utils.normalize)
    positions.to_excel('../data/dedup/solr_positions.xlsx', index=False)
    (positions['i'].value_counts()/positions.shape[0]).head(40)

    # ax = positions.plot(kind='hist', title='positions')
    # fig = ax.get_figure()
    # fig.savefig('../data/dedup/solr_positions.pdf')


if __name__ == "__main__":
    pass
    # solr_stat()
