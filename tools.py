from absl import flags
from absl import app
from glob import glob
import json
import re
import pickle
import pandas as pd
import numpy as np
import zipfile
import os
from textblob.tokenizers import WordTokenizer
import string
from nltk.stem.snowball import SnowballStemmer
import cyrtranslit
import paramiko
import time
from sklearn.externals import joblib
from pprint import pprint
from tqdm import tqdm
from pymongo import MongoClient

c_HOST = '10.70.6.154'
ml_HOST = '10.72.102.67'

FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir", None, "data directory")
flags.DEFINE_string("feed_db", '1cfreshv4', "feed mongodb database name")
flags.DEFINE_string("release_db", 'release', "master mongodb database name")
flags.DEFINE_string("mongo_host", c_HOST, "MongoDb host")
flags.DEFINE_string("solr_host", ml_HOST, "SOLR host")
flags.DEFINE_bool("notranslit", False,
                  "don't transliterate english to cyrillic")
flags.DEFINE_bool("build_tfidf", False, "build tfidf model")
flags.DEFINE_bool("tfidf", False, "use tfidf features")
flags.DEFINE_bool("fasttext", False, "use fasttext features")
flags.DEFINE_bool("for_test", False, "sample just for test")
flags.DEFINE_bool("build_features", False, "build column features")
flags.DEFINE_bool("build_fasttext", False, "build fasttext features")
flags.DEFINE_bool("build_tfrecord", False,
                  "build tensorflow record input files")
flags.DEFINE_integer("nrows", 100, "The TOP number of rows to query")


chars = string.punctuation
chars = chars.replace('%', '').replace('_', '').replace('@', '')
prog = re.compile("[\\W\\d]", re.UNICODE)
prog_with_digits = re.compile("[\\W]", re.UNICODE)

float_prog = re.compile(r"[-+]?\d*\.\d+|\d+", re.UNICODE)
dot_prog = re.compile(r'[xх*]', re.UNICODE)

TransTable = str.maketrans(dict.fromkeys(r'~/-\[\]()|{}:^+', ' '))
wt = WordTokenizer()

unit_lookup = {
    'г': 'грамм', 'грам': 'грамм', 'гр': 'грамм', 'грамм': 'грамм', 'gr': 'грамм',
    'ml': 'мл', 'милл': 'мл', 'млитр': 'мл', 'млтр': 'мл', 'мл': 'мл',
    'ш': 'шт', 'шт': 'шт',
    'тон': 'тонна', 'тн': 'тонна', 'тонна': 'тонна', 'тонн': 'тонна',
    'л': 'литр', 'литр': 'литр', 'лит': 'литр',
    'kg': 'кг', 'кг': 'кг',
    'mm': 'мм', 'cm': 'см', 'мм': 'мм', 'см': 'см', 'дм': 'дм',
    '№': 'номер', 'номер': 'номер',
    'ват': 'ватт', 'вт': 'ватт', 'ватт': 'ватт'}

stemmer = SnowballStemmer("russian", ignore_stopwords=True)


def normalize(sent, stem=False, translit=True):
    """
    This works good but slow, redo
    """
    tokens = normalize_v2(sent, translit)
    if stem:
        tokens = (stemmer.stem(t) for t in tokens)
    sent = " ".join((t for t in tokens))
    return sent


def isnum(t):
    try:
        f = float(t)
        # if f.is_integer():
        #     return str(int(f))
        return str(f)
    except:
        return False


def split_unit(t):
    if len(t) == 0 or not t[0].isdigit():
        return t
    tmp = float_prog.findall(t)
    if len(tmp):
        striped = t.lstrip(tmp[0])
        if len(tmp) > 1:
            ix = striped.find(tmp[1])
            postfix = striped[:ix]
        else:
            postfix = striped

        return str(float(tmp[0])), unit_lookup.get(postfix, postfix)
    return t


def proceed_token(t, translit=False):
    t = t.replace('ё', 'е').replace('й', 'и').replace(',', '.')
    num = isnum(t)
    if num:
        return num

    t = t.rstrip('ъ')

    # # all ascii
    if translit and all(ord(char) < 128 for char in t):
        t = cyrtranslit.to_cyrillic(t, 'ru')

    tmp = dot_prog.split(t)
    if len(tmp) > 1:
        tmp = [isnum(el) for el in tmp]
        if all(tmp):
            return 'x'.join(tmp)  # english x

    tmp = split_unit(t)
    if type(tmp) == tuple:
        return tmp[0] + ' ' + tmp[1]

    t = t.replace('.', ' ')
    tmp = (unit_lookup.get(t, t) for t in t.split(' '))
    t = ' '.join((ti for ti in tmp if len(ti) > 1))
    return t


def normalize_v2(sent, translit=False):
    tmp = sent.translate(TransTable).lower()
    tokens = (proceed_token(t, translit) for t in wt.tokenize(tmp, False))
    return tokens


def current_dir():
    import inspect

    filename = inspect.getframeinfo(inspect.currentframe()).filename
    path = os.path.dirname(os.path.abspath(filename))
    return path


def load_master():
    dir_name = os.path.join(current_dir(), '../data/master/')

    pkls = glob(dir_name + 'master**.pkl')
    if len(pkls):
        data = do_unpickle(pkls[0])
    else:
        fname = sorted(glob(dir_name + 'master**.zip'))[-1]
        with zipfile.ZipFile(fname, 'r') as z:
            inner_name = z.namelist()[0]
            with z.open(inner_name) as f:
                text = f.read()
            data = json.loads(text.decode("utf-8"))
        do_pickle(data, dir_name + 'master.pkl')
    print('master loaded')
    return Updater(data)


def do_pickle(data, fname):
    with open(fname, 'wb') as f:
        joblib.dump(data, f)


def do_unpickle(fname):
    with open(fname, 'rb') as f:
        data = joblib.load(f)
    return data


def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


class Updater:
    def __init__(self, master):
        self.ets = []
        self.id2cat, self.cat2id, self.id2man, \
            self.man2id, self.brand2id, \
            self.id2et, self.ftr2id, \
            self.id2brand, self.id2clr, self.fval2id = [dict()] * 10

        self.db = master['Database']
        if type(self.db) == list:
            self.db = self.db[0]
        self.tags = {'etalons': 'ets',
                     'features': 'ftrs',
                     'categories': 'cats',
                     'featureValues': 'fvals',
                     'brands': 'brands',
                     'subBrands': 'subBrands',
                     'goodTypes': 'goodTypes',
                     'manufacturers': 'mans',
                     'suppliers': 'sups',
                     'units': 'units',
                     'classifiers': 'clrs'}

        self.attr_cols = ['barcodes', 'synonyms', 'manufacturerId', 'comment',
                          'brandId', 'description', 'manufacturerCode',
                          'source', 'srcId', 'unitName', 'classifiers',
                          'weight', 'volume', 'length', 'nameShort',
                          'unitOKEI', 'vat', 'area', 'subBrandId',
                          'goodTypeId', 'supplierId']

        for k, v in self.tags.items():
            setattr(self, v, self.db.get(k, {}))

        for k, v in self.tags.items():
            ltag = 'name' if v in {'units', 'clrs'} else 'id'
            name = 'id2' + v.rstrip('s')
            setattr(self, name, self.get_dict(getattr(self, v), ltag))

        for k, v in self.tags.items():
            if v in {'units', 'clrs'}:
                continue
            elif v == 'fvals':
                ltag = 'value'
            elif v == 'ftrs':
                ltag = 'name'
            name = v.rstrip('s') + '2id'
            setattr(self, name, self.get_dict(getattr(self, v), ltag, 'id'))

    @staticmethod
    def get_dict(array, ltag, rtag=None):
        if not array:
            return None
        elif rtag:
            return {el[ltag]: el[rtag] for el in array if ltag in el}
        else:
            return {el[ltag]: el for el in array if ltag in el}


def unique_syn(syn):
    unique = set()
    res = []
    for tag in syn.split():
        if tag not in unique:
            res.append(tag)
            unique.update([tag])
    return ' '.join(res)


def constitute_text(name, et, id2brand, use_syns=False):
    text = name.lower()
    if use_syns:
        syns = [s['name'].lower() for s in et.get('synonyms', [])]
        if syns:
            syns = [text] + syns
            text = unique_syn(' '.join(syns))

    bid = et.get('brandId')
    if bid:
        bname = id2brand[bid]['name'].lower()
        if bname not in text:
            text += ' ' + bname
    return text


def load_samples(filename, key='samples'):
    npzfile = np.load(filename)
    samples = pd.DataFrame(npzfile[key])
    if len(samples) == 0:
        return None
    samples.columns = npzfile['columns']
    return samples


def scp(host, user, localfile, remotefile):
    transport = paramiko.Transport((host, 22))
    transport.connect(username=user, pkey='~/.ssh/id_rsa')
    sftp = paramiko.SFTPClient.from_transport(transport)
    sftp.put(localfile, remotefile)
    sftp.close()
    transport.close()


def feed2mongo(feed, dbname):
    client = MongoClient()
    client.drop_database(dbname)
    db = client[dbname]

    fdb = feed['Database']
    if isinstance(fdb, list):
        fdb = fdb[0]

    for k, v in fdb.items():
        if not isinstance(v, list):
            continue
        for el in v:
            _id = el.pop('id', None)
            if _id is not None:
                el['_id'] = _id

        if len(v):
            db[k].insert_many(v)
