from fuzzywuzzy import fuzz
import jellyfish
import textdistance as td
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.optimal_string_alignment import OptimalStringAlignment
from similarity.damerau import Damerau
from similarity.metric_lcs import MetricLCS
from similarity.ngram import NGram
from similarity.qgram import QGram
import py_stringmatching as sm

q = "мак кофе премиум субл  пак 75.0 грамм"
d = "кофе маккофе premium  пак 75.0 грх"

me = sm.MongeElkan()
menlw = sm.MongeElkan(sim_func=sm.NeedlemanWunsch().get_raw_score)
qgram = QGram(2)
twogram = NGram(2)
threegram = NGram(3)
fourgram = NGram(4)
damerau = Damerau()
metric_lcs = MetricLCS()
osa = OptimalStringAlignment()


def get_sim_features():
    q_split = q.split()
    d_split = d.split()

    ftrs = {'q_len': len(q), 'd_len': len(d), 'q/d': len(q)/len(d)}
    ftrs['difflib.ratio'] = fuzz.ratio(q, d)/100
    ftrs['fuzz.partial_ratio'] = fuzz.partial_ratio(q, d)/100
    ftrs['fuzz.token_sort_ratio'] = fuzz.token_sort_ratio(q, d)/100
    ftrs['fuzz.token_set_ratio'] = fuzz.token_set_ratio(q, d)/100

    ftrs['jellyfish.jaro_winkler'] = jellyfish.jaro_winkler(q, d)

    ftrs['hamming'] = td.hamming.normalized_distance(q, d)
    ftrs['mlipns'] = td.mlipns.normalized_distance(q, d)
    ftrs['levenshtein'] = td.levenshtein.normalized_distance(q, d)
    ftrs['needleman_wunsch'] = td.needleman_wunsch.normalized_distance(q, d)
    ftrs['gotoh'] = td.gotoh.normalized_distance(q, d)
    ftrs['jaccard'] = td.jaccard.normalized_distance(q, d)
    ftrs['sorensen_dice'] = td.sorensen_dice.normalized_distance(q, d)
    ftrs['tversky'] = td.tversky.normalized_distance(q, d)
    ftrs['overlap'] = td.overlap.normalized_distance(q, d)
    ftrs['cosine'] = td.cosine.normalized_distance(q, d)
    ftrs['bag'] = td.bag.normalized_distance(q, d)
    ftrs['lcsstr'] = td.lcsstr.normalized_distance(q, d)
    ftrs['ratcliff'] = td.ratcliff_obershelp.normalized_distance(q, d)
    ftrs['bz2_ncd'] = td.bz2_ncd(q, d)
    ftrs['lzma_ncd'] = td.lzma_ncd.normalized_distance(q, d)
    ftrs['rle_ncd'] = td.rle_ncd.normalized_distance(q, d)
    ftrs['bwtrle_ncd'] = td.bwtrle_ncd.normalized_distance(q, d)
    ftrs['zlib_ncd'] = td.zlib_ncd.normalized_distance(q, d)
    ftrs['mra'] = td.mra.normalized_distance(q, d)
    ftrs['editex'] = td.editex.normalized_distance(q, d)
    ftrs['prefix'] = td.prefix.normalized_distance(q, d)
    ftrs['postfix'] = td.postfix.normalized_distance(q, d)
    ftrs['length'] = td.length.normalized_distance(q, d)
    ftrs['identity'] = td.identity.normalized_distance(q, d)
    ftrs['matrix'] = td.matrix.normalized_distance(q, d)

    ftrs['damerau'] = damerau.distance(q, d)
    ftrs['osa'] = osa.distance(q, d)
    ftrs['metric_lcs'] = metric_lcs.distance(q, d)
    ftrs['twogram'] = twogram.distance(q, d)
    ftrs['threegram'] = threegram.distance(q, d)
    ftrs['fourgram'] = fourgram.distance(q, d)
    ftrs['qgram'] = qgram.distance(q, d)
    ftrs['me.get_raw_score'] = me.get_raw_score(q_split, d_split)
    ftrs['menlw'] = menlw.get_raw_score(q_split, d_split)

    return ftrs
