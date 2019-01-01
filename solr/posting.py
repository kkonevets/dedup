import utils
import json
import io
from tqdm import tqdm
import string


def copy_key(src, dst, key):
    val = src.get(key)
    if val:
        dst[key] = val


def clean_et(et):
    for k, v in et.items():
        if type(v) == str:
            et[k] = utils.normalize(v)


def etalons_to_docs():
    master = utils.load_master()
    mdb = master['Database']

    id2cat = {c['id']: c for c in mdb['categories']}
    id2brand = {c['id']: c for c in mdb['brands']}

    ets = []
    for et in tqdm(mdb['etalons']):
        cid = et['categoryId']

        new_et = {
            'id': et['id'],
            'name': et['name'],
            'categoryId': cid,
            'categoryName': id2cat[cid]['name']
        }

        copy_key(et, new_et, 'barcodes')
        copy_key(et, new_et, 'unitName')
        copy_key(et, new_et, 'source')
        # copy_key(et, new_et, 'manufacturerCode')
        # copy_key(et, new_et, 'brandId')
        # copy_key(et, new_et, 'manufacturerId')
        # copy_key(et, new_et, 'description')

        bid = et.get('brandId')
        if bid:
            new_et['brand'] = id2brand[bid]['name']

        clean_et(new_et)
        ets.append(new_et)

    with io.open('../data/solr/master_data.json', 'w', encoding='utf8') as f:
        json.dump(ets, f, ensure_ascii=False)


if __name__ == '__main__':
    pass
    etalons_to_docs()
