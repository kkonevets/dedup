from glob import glob
import json
import pickle
import zipfile
import os
from textblob.tokenizers import WordTokenizer

TransTable = str.maketrans(dict.fromkeys('/-()|{}', ' '))
wt = WordTokenizer()


def normalize(sent):
    sent = sent.translate(TransTable).lower()
    tokens = (t for t in wt.tokenize(sent, False) if len(t) > 1)
    sent = " ".join(tokens)
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
        with open(pkls[0], 'rb') as f:
            data = pickle.load(f)
    else:
        fname = sorted(glob(dir_name + 'master**.zip'))[-1]
        with zipfile.ZipFile(fname, 'r') as z:
            inner_name = z.namelist()[0]
            with z.open(inner_name) as f:
                text = f.read()
            data = json.loads(text.decode("utf-8"))
        with open(dir_name + 'master.pkl', 'wb') as f:
            pickle.dump(data, f)
    print('master loaded')
    return data
