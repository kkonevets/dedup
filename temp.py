import sklearn
import io
import json
from tqdm import tqdm
import tools
import pandas as pd
import numpy as np
from pymongo import MongoClient
import fuzzy
import h5py
from functools import lru_cache


with io.open('../data/dedup/1cnrel-2019-02-26-19.json', encoding='utf8') as f:
    feed = json.load(f)

tools.feed2mongo(feed, 'release2')
