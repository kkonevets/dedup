import tools


class Producer:
    def __init__(self, data_dir, colnames):
        self.data_dir = data_dir
        self.colnames = colnames
        self.load_data()

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
        for _id, text in corpus[corpus[tag].isin(ids)][[tag, 'text']].values:
            id2text[_id] = text
        return id2text

    def gen_data(self, cur_samples):
        fid2text, sid2text = self.fid2text, self.sid2text
        for row in tools.tqdm(cur_samples.itertuples(), total=len(cur_samples)):
            q_splited = sid2text[row.synid].split()
            d_splited = fid2text[row.fid].split()

            if row.target == 0 and ' '.join(d_splited) == ' '.join(q_splited):
                continue

            if len(q_splited) * len(d_splited) == 0:
                continue

            # TODO: add DNN features: brands ...
            yield q_splited, d_splited, [getattr(row, c) for c in self.colnames]

    def load_data(self):
        samples = tools.load_samples(self.data_dir + '/samples.npz')

        # exclude samples not found in TOP
        synids_exclude = set(samples[samples['ix'] == -1]['synid'].unique())
        samples = samples[~samples['synid'].isin(synids_exclude)]

        sids = samples[samples['synid'] != -1]['synid'].unique()
        fids = samples['fid'].unique()

        corpus = tools.load_samples(self.data_dir + '/corpus.npz')

        sid2text = self.get_id2text(corpus, 'synid', sids)
        fid2text = self.get_id2text(corpus, 'fid', fids)

        # vals = corpus[corpus['train'] != 0]['text'].values
        # informative_terms = set([w for s in vals for w in s.split()])
        # with io.open(FLAGS.data_dir + '/vocab.txt', 'w', encoding='utf8') as f:
        #     for term in informative_terms:
        #         f.write(term + '\n')

        self.samples, self.sid2text, self.fid2text = \
            samples, sid2text, fid2text
