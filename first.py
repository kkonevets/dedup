import json
import urllib
import string
from tqdm import tqdm
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
        if bid:
            bname = id2brand[bid]['name']
            if bname.lower() not in text:
                text += ' ' + bname

        text = text.translate(tt).lower()
        found = query_solr(text, nrows)
        if len(found) == 0:
            positions.append(-1)
        for i, el in enumerate(found):
            curbcs = [int(c) for c in el.get('barcodes', [])]
            if len(bcs.intersection(curbcs)):
                positions.append(i)
                break
