from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import jellyfish
import textdistance
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
optimal_string_alignment = OptimalStringAlignment()


def get_sim_features():
    m = SequenceMatcher(None, q, d)
    q_split = q.split()
    d_split = d.split()

    ftrs = {'q_len': len(q), 'd_len': len(d), 'q/d': len(q)/len(d)}
    ftrs['difflib.ratio'] = m.ratio()
    ftrs['fuzz.partial_ratio'] = fuzz.partial_ratio(q, d)/100
    ftrs['fuzz.token_sort_ratio'] = fuzz.token_sort_ratio(q, d)/100
    ftrs['fuzz.token_set_ratio'] = fuzz.token_set_ratio(q, d)/100

    ftrs['jellyfish.jaro_winkler'] = jellyfish.jaro_winkler(q, d)

    ftrs['hamming'] = \
        textdistance.hamming.normalized_distance(q, d)
    ftrs['mlipns'] = \
        textdistance.mlipns.normalized_distance(q, d)
    ftrs['levenshtein'] = \
        textdistance.levenshtein.normalized_distance(q, d)
    ftrs['needleman_wunsch'] = \
        textdistance.needleman_wunsch.normalized_distance(q, d)
    ftrs['gotoh'] = \
        textdistance.gotoh.normalized_distance(q, d)
    ftrs['jaccard'] = \
        textdistance.jaccard.normalized_distance(q, d)
    ftrs['sorensen_dice'] = \
        textdistance.sorensen_dice.normalized_distance(q, d)
    ftrs['tversky'] = \
        textdistance.tversky.normalized_distance(q, d)
    ftrs['overlap'] = \
        textdistance.overlap.normalized_distance(q, d)
    ftrs['cosine'] = \
        textdistance.cosine.normalized_distance(q, d)
    ftrs['bag'] = \
        textdistance.bag.normalized_distance(q, d)
    ftrs['lcsstr'] = \
        textdistance.lcsstr.normalized_distance(q, d)
    ftrs['ratcliff_obershelp'] = \
        textdistance.ratcliff_obershelp.normalized_distance(q, d)
    ftrs['bz2_ncd'] = \
        textdistance.bz2_ncd(q, d)
    ftrs['lzma_ncd'] = \
        textdistance.lzma_ncd.normalized_distance(q, d)
    ftrs['rle_ncd'] = \
        textdistance.rle_ncd.normalized_distance(q, d)
    ftrs['bwtrle_ncd'] = \
        textdistance.bwtrle_ncd.normalized_distance(q, d)
    ftrs['zlib_ncd'] = \
        textdistance.zlib_ncd.normalized_distance(q, d)
    ftrs['mra'] = \
        textdistance.mra.normalized_distance(q, d)
    ftrs['editex'] = \
        textdistance.editex.normalized_distance(q, d)
    ftrs['prefix'] = \
        textdistance.prefix.normalized_distance(q, d)
    ftrs['postfix'] = \
        textdistance.postfix.normalized_distance(q, d)
    ftrs['length'] = \
        textdistance.length.normalized_distance(q, d)
    ftrs['identity'] = \
        textdistance.identity.normalized_distance(q, d)
    ftrs['matrix'] = \
        textdistance.matrix.normalized_distance(q, d)

    ftrs['damerau'] = damerau.distance(q, d)
    ftrs['optimal_string_alignment'] = optimal_string_alignment.distance(q, d)
    ftrs['metric_lcs'] = metric_lcs.distance(q, d)
    ftrs['twogram'] = twogram.distance(q, d)
    ftrs['threegram'] = threegram.distance(q, d)
    ftrs['fourgram'] = fourgram.distance(q, d)
    ftrs['qgram'] = qgram.distance(q, d)
    ftrs['me.get_raw_score'] = me.get_raw_score(q_split, d_split)
    ftrs['menlw'] = menlw.get_raw_score(q_split, d_split)

    return ftrs
