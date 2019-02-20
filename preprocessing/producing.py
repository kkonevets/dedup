import tools
import pandas as pd
from dataclasses import dataclass


@dataclass
class QDInfo:
    q_terms: list
    d_terms: list
    ixs: list


class Producer:
    def __init__(self, data_dir, colnames):
        self.data_dir = data_dir
        self.colnames = colnames
        self._load_data()

    def gen_pairs(self):
        samples = self.samples
        if 'train' in samples.columns:
            train_gen = self.gen_data(
                samples[samples['train'] == 1])
            test_samples = samples[samples['train'] == 0]
        else:
            train_gen = iter(())
            test_samples = samples

        test_gen = self.gen_data(test_samples)
        return train_gen, test_gen

    @staticmethod
    def get_id2text(corpus, tag, ids):
        id2text = {}
        temp = corpus[corpus[tag].isin(ids)]
        for _id, text in temp[[tag, 'text']].values:
            id2text[_id] = text
        return id2text

    def gen_data(self, cur_samples):
        qid2text, fid2text, sid2text = self.qid2text, self.fid2text, self.sid2text
        for row in tools.tqdm(cur_samples.itertuples(), total=len(cur_samples)):
            if pd.notna(row.synid):
                q_terms = sid2text[row.synid].split()
            else:
                q_terms = qid2text[row.qid].split()
            d_terms = fid2text[row.fid].split()

            if row.target == 0 and ' '.join(d_terms) == ' '.join(q_terms):
                continue

            if len(q_terms) * len(d_terms) == 0:
                continue

            qdi = QDInfo(q_terms=q_terms,
                         d_terms=d_terms,
                         ixs=[getattr(row, c) for c in self.colnames])

            # TODO: add DNN features: brands ...
            yield qdi

    def _load_data(self):
        samples = tools.load_samples(self.data_dir + '/samples.npz')
        corpus = tools.load_samples(self.data_dir + '/corpus.npz')

        qid2text = self.get_id2text(corpus, 'qid', samples['qid'].unique())
        sid2text = self.get_id2text(corpus, 'synid', samples['synid'].unique())
        fid2text = self.get_id2text(corpus, 'fid', samples['fid'].unique())

        self.samples, self.qid2text, self.sid2text, self.fid2text = \
            samples, qid2text, sid2text, fid2text
