import json
import urllib
import string
from tqdm import tqdm
import pandas as pd
from urllib.parse import quote


def query_solr(text, rows=1, exclude=[]):
    q = 'http://c:8983/solr/nom_core/select?df=my_text_ru' \
        '&q=%s&rows=%d' % (quote(text), rows)
    r = urllib.request.urlopen(q).read()
    docs = json.loads(r)['response']['docs']
    return docs


def solr_stat():
    with open('../data/1cfresh/1cfreshv4.json', 'r') as f:
        fresh = json.load(f)

    db = fresh['Database']
    singles = []
    for et in db['etalons']:
        if '#single' in et['comment']:
            singles.append(et)

    id2brand = {b['id']: b for b in db['brands']}
    positions = []
    nrows = 100
    tt = str.maketrans(dict.fromkeys(string.punctuation, ' '))

    for et in tqdm(singles):
        bcs = set([int(c) for c in et['barcodes']])
        text = et['name']
        bid = et.get('brandId')
        bname = ''
        if bid:
            bname = id2brand[bid]['name']
            if bname.lower() not in text:
                text += ' ' + bname

        text = text.translate(tt).lower()
        found = query_solr(text, nrows)

        rec = [et['id'], et['name'], bname]
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
        lambda s: str.lower) == positions['el_name'].apply(lambda s: str.lower)
    positions.to_excel('../data/dedup/solr_positions.xlsx', index=False)
    positions['i'].value_counts()/positions.shape[0]
    # ax = positions.plot(kind='hist', title='positions')
    # fig = ax.get_figure()
    # fig.savefig('../data/dedup/solr_positions.pdf')
