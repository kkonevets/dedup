import io
import json
from tqdm import tqdm
import tools
import pandas as pd
import numpy as np
from pymongo import MongoClient
import fuzzy
import h5py

ftrs = tools.load_samples('../data/dedup/corpus.npz')
ftrs = ftrs[['synid', 'fid', 'train']].values.astype(np.float)


def append_h5(fname, vals, columns):
    vals = np.array(vals, dtype=np.float32)
    with h5py.File(fname, 'a') as hf:
        if len(hf.keys()) == 0:
            hf.create_dataset('ftrs', data=vals,
                              maxshape=(None, vals.shape[1]),
                              dtype='f', chunks=True)
        else:
            hf['ftrs'].resize(hf['ftrs'].shape[0] + vals.shape[0], axis=0)

        hf['ftrs'][-vals.shape[0]:] = vals
        hf['ftrs'].attrs['columns'] = columns


append_h5('../data/dedup/50_50_ftrs.h5', ftrs, ['synid', 'fid', 'train'])


hf = h5py.File('../data/dedup/50_50_ftrs.h5', 'a')
hf['ftrs'].attrs['columns']
hf.close()
