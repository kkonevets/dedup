import matplotlib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
import numpy as np
import pandas as pd
import tools
from collections import Counter

matplotlib.use('agg')


def plot_topn_curves(positions_list, fname, scale=1, labels=None, title=None):
    if labels is not None:
        assert len(positions_list) == len(labels)
    else:
        labels = [None]*len(positions_list)

    import matplotlib.pyplot as plt
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
    ax = df.plot(title=title, grid=True)
    # xtics = list(df.index)[::2]
    # if xtics[-1] != df.index[-1]:
    #     xtics.append(df.index[-1])
    # ax.set_xticks(df.index)
    fig = ax.get_figure()
    ax.set_xlabel("top N")
    ax.set_ylabel("recall")
    fig.savefig(fname)


def plot_precision_recall(y_true, probas_pred, tag='', recall_scale=1):
    import matplotlib.pyplot as plt

    average_precision = average_precision_score(y_true, probas_pred)
    precision, recall, _ = precision_recall_curve(y_true, probas_pred)

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
    ax.set_title('Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))
    fig.savefig('../data/dedup/prec_recal%s.pdf' % tag, dpi=400)


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
