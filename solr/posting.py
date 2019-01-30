import tools
import json
import io
from tqdm import tqdm
import re
import multiprocessing as mp
from pymongo import MongoClient

client = MongoClient()
db = client['master']

id2cat = {c['_id']: c for c in db.categories.find({}, projection=['name'])}
id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}

prog = re.compile("[\\W]", re.UNICODE)


def copy_key(src, dst, key):
    val = src.get(key)
    if val:
        dst[key] = val


def clean_et(et, exclude=None):
    if exclude is None:
        keys = et.keys()
    else:
        keys = set(et.keys()) - set(exclude)

    for k in keys:
        v = et[k]
        if type(v) == str:
            et[k] = tools.normalize(v)


def unique_syn(syn):
    unique = set()
    res = []
    for tag in syn.split():
        if tag not in unique:
            res.append(tag)
            unique.update([tag])
    return ' '.join(res)


def proceed_one(et):
    cid = et['categoryId']

    new_et = {
        'id': et['_id'],
        'name': et['name'],
        'categoryId': cid,
        'categoryName': id2cat[cid]['name']
    }

    copy_key(et, new_et, 'barcodes')
    copy_key(et, new_et, 'source')
    # copy_key(et, new_et, 'manufacturerCode')
    # copy_key(et, new_et, 'brandId')
    # copy_key(et, new_et, 'manufacturerId')
    # copy_key(et, new_et, 'description')

    un = et.get('unitName')
    if un:
        new_et['unitName'] = ' '.join(prog.sub(' ', un).lower().split())

    syns = [s['name'] for s in et.get('synonyms', [])]
    if syns:
        new_et['synonyms'] = '\n'.join(syns)

    bid = et.get('brandId')
    if bid:
        new_et['name'] += ' ' + id2brand[bid]['name']

    clean_et(new_et, exclude=['source', 'unitName'])

    sval = new_et.get('synonyms')
    if sval:
        new_et['synonyms'] = unique_syn(sval)

    return new_et


def etalons_to_docs():
    total = db.etalons.count_documents({})
    iterator = db.etalons.find(
        {}, projection=['name', 'categoryId', 'synonyms', 'brandId',
                        'barcodes', 'unitName', 'source'])

    ets = []
    with mp.Pool(mp.cpu_count()) as p:  # maxtasksperchild=100000
        with tqdm(total=total) as pbar:
            for new_et in tqdm(p.imap_unordered(proceed_one, iterator)):
                ets.append(new_et)
                pbar.update()

    with io.open('../data/solr/master_data.json', 'w', encoding='utf8') as f:
        json.dump(ets, f, ensure_ascii=False)


if __name__ == '__main__':
    pass
    # etalons_to_docs()
