from glob import glob
import json
import re
import pickle
import zipfile
import os
from textblob.tokenizers import WordTokenizer
import string

chars = string.punctuation
chars = chars.replace('%', '').replace('_', '').replace('@', '')
punct_tt = str.maketrans(dict.fromkeys(chars, ' '))
prog = re.compile("[\\W\\d]", re.UNICODE)

float_prog = re.compile(r"[-+]?\d*\.\d+|\d+", re.UNICODE)

TransTable = str.maketrans(dict.fromkeys(r'/-()|{}:^+', ' '))
wt = WordTokenizer()

unit_lookup = {'г': 'грамм', 'грам': 'грамм', 'гр': 'грамм', 'ml': 'мл',
               'милл': 'мл', 'млитр': 'мл', 'млтр': 'мл',
               'ш': 'шт', 'л': 'литр', 'kg': 'кг', 'mm': 'мм', 'cm': 'см'}


def normalize(sent):
    return normalize_v1(sent)


def isnum(t):
    try:
        return str(float(t))
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


def proceed_token(t):
    t = t.replace('ё', 'е').replace(',', '.')
    num = isnum(t)
    if num:
        return num

    tmp = t.split('х')  # russian х
    if len(tmp) == 1:
        tmp = t.split('*')
    if len(tmp) > 1 and isnum(tmp[0]) and isnum(tmp[1]):
        return 'x'.join(tmp)  # english x

    t = split_unit(t)
    if type(t) == tuple:
        return t[0] + ' ' + t[1]

    t = t.replace('.', ' ')
    tmp = (unit_lookup.get(t, t) for t in t.split(' '))
    t = ' '.join((ti for ti in tmp if len(ti) > 1))
    return t


def normalize_v2(sent):
    tmp = sent.translate(TransTable).lower()
    tokens = (proceed_token(t) for t in wt.tokenize(tmp, False))
    tmp = " ".join((t for t in tokens if len(t) > 1))
    return tmp


def normalize_v1(sent):
    """
    this is better than v0
    """
    sent = sent.translate(TransTable).lower()
    tokens = (t for t in wt.tokenize(sent, False) if len(t) > 1)
    sent = " ".join(tokens)
    return sent


def normalize_v0(sent):
    tokens = sent.translate(punct_tt).lower().split()
    tokens = (prog.sub('', w) for w in tokens)
    sent = ' '.join((w for w in tokens if len(w) > 1))
    return sent


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
    return data


def do_pickle(data, fname):
    with open(fname, 'wb') as f:
        pickle.dump(data, f)


def do_unpickle(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data
