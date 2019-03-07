import matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import tools
import seaborn as sns
from collections import Counter
from preprocessing.letor import INFO_COLUMNS
import matplotlib.pyplot as plt

matplotlib.use('agg')


def plot_binary_prob_freqs(y_test, probs):
    ones = probs[np.where(y_test == 1)]
    zeros = probs[np.where(y_test == 0)]

    plt.clf()

    ax = sns.distplot(ones, label='1', norm_hist=True)
    sns.distplot(zeros, label='0', norm_hist=True, ax=ax)
    ax.set_xlabel('prob')
    ax.set_ylabel('density')
    ax.set_ylim([0.0, 15])
    ax.legend(loc="best")

    fig = ax.get_figure()
    fig.savefig('../data/dedup/prob_freqs.pdf', dpi=400)


def plot_topn_curves(positions_list, fname, scale=1, labels=None, title=None):
    if labels is not None:
        assert len(positions_list) == len(labels)
    else:
        labels = [None]*len(positions_list)

    plt.clf()

    csums = []
    for positions in positions_list:
        rel1 = (positions.value_counts()/len(positions)).head(40)
        print(rel1)

        psex = positions[~positions.isin([-1, -2])]
        rel2 = (psex.value_counts()/len(positions))
        print(rel2.sum())
        print('\n')

        cumsum = rel2.sort_index().cumsum()
        cumsum.index += 1
        csums.append(cumsum)

    df = pd.DataFrame()
    for cumsum, label in zip(csums, labels):
        df[label] = cumsum

    df *= scale

    title = title if title else 'found in top N'
    ax = df.plot(title=title, grid=True, xlim=(-0.5, df.index.max()))
    # xtics = list(df.index)[::2]
    # if xtics[-1] != df.index[-1]:
    #     xtics.append(df.index[-1])
    # ax.set_xticks(df.index)
    fig = ax.get_figure()
    ax.set_xlabel("top N")
    ax.set_ylabel("recall")
    fig.savefig(fname)


def plot_precision_recall_straight(precision, recall, 
            average_precision=None, tag='', recall_scale=1, prefix=''):
    import matplotlib.pyplot as plt

    precision = np.array(precision)
    recall = np.array(recall)

    plt.clf()
    fig, ax = plt.subplots()

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'})
    ax.step(recall*recall_scale, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall*recall_scale, precision, alpha=0.2, color='b',
                    rasterized=True, **step_kwargs)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1])
    ax.set_xlim([0.0, 1])
    if average_precision:
        ax.set_title(prefix+'Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))
    fig.savefig('../data/dedup/prec_recal%s.pdf' % tag, dpi=400)


def plot_precision_recall(y_true, probas_pred, tag='', recall_scale=1):
    average_precision = average_precision_score(y_true, probas_pred)
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)

    plot_precision_recall_straight(precision, recall, 
        tag=tag, recall_scale=recall_scale, 
        average_precision=average_precision)

 

def get_recall_test_scale():
    samples = tools.load_samples('../data/dedup/samples.npz')
    ptest = samples[(samples['train'] == 0) & (samples['target'] == 1)]
    counts = ptest['ix'].value_counts()
    recall_scale = 1-counts[-1]/counts.sum()
    return recall_scale


def ranker_predict(ranks, dmatrix, groups):
    y = dmatrix.get_label()
    positions = []
    ix_prev = 0
    scores = {i: [] for i in [1, 2, 5, 6, 7, 10]}
    for gcount in tools.tqdm(groups):
        y_cur = y[ix_prev: ix_prev + gcount]
        r = ranks[ix_prev: ix_prev + gcount]
        rsorted = y_cur[np.argsort(r)[::-1]]
        ix = np.where(rsorted == 1)[0][0]
        positions.append(ix)
        for k in scores.keys():
            val = ndcg_at_k(rsorted, k, method=1)
            scores[k].append(val)
        ix_prev += gcount

    for k in list(scores.keys()):
        scores['ndcg@%d' % k] = np.round(np.mean(scores.pop(k)), 4)

    positions = pd.Series(positions)
    return scores, positions


def clr_predict(probs, dmtx, threshold=0.4):
    y = dmtx.get_label()
    c = Counter(y)
    y_pred = (probs >= threshold).astype(int)
    rep = classification_report(y, y_pred, labels=[1], output_dict=True)
    rep = rep['1']
    rep['base_accuracy'] = c[0]/sum(c.values())
    rep['accuracy'] = accuracy_score(y, y_pred)
    rep = {k: round(v, 4) for k, v in rep.items()}
    tools.pprint(rep)
    print('\n')
    return y_pred


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def examples_to_view(ftest, feed_db, release_db):
    from pymongo import MongoClient

    test_probs = ftest['prob']

    client = MongoClient(tools.c_HOST)
    db = client[feed_db]
    mdb = client[release_db]

    columns = INFO_COLUMNS + ['prob']
    cond = (test_probs > 0.9) & (ftest['target'] == 0)
    df1 = ftest[cond][columns].copy()

    cond = (test_probs < 0.1) & (ftest['target'] == 1)
    df2 = ftest[cond][columns].copy()

    qids = set(df1['qid'].unique())
    qids.update(df2['qid'].unique())
    fids = set(df1['fid'].unique())
    fids.update(df2['fid'].unique())

    qid2et = {et['_id']: et for et in db.etalons.find(
        {'_id': {'$in': list(qids)}})}
    fid2et = {et['_id']: et for et in mdb.etalons.find(
        {'_id': {'$in': list(fids)}})}
    id2brand = {c['_id']: c for c in db.brands.find({}, projection=['name'])}
    mid2brand = {c['_id']: c for c in mdb.brands.find({}, projection=['name'])}

    for df in (df1, df2):
        qs, ds = [], []
        for row in df.itertuples():
            et = qid2et[row.qid]
            if pd.isna(row.synid):
                name = next((s['name'] for s in et.get('synonyms')
                            if s['id'] == row.synid))
            else:
                name = et['name']
            q = tools.constitute_text(
                name, et, id2brand, use_syns=False)

            met = fid2et[row.fid]
            d = tools.constitute_text(
                met['name'], met, mid2brand, use_syns=True)
            qs.append(q)
            ds.append(d)
        df['q'] = qs
        df['d'] = ds
        df = df[columns + ['q', 'd']]

    df1.sort_values('prob', ascending=False, inplace=True)
    df2.sort_values('prob', ascending=True, inplace=True)
    df1.to_excel('../data/dedup/typeI.xlsx', index=False)
    df2.to_excel('../data/dedup/typeII.xlsx', index=False)
