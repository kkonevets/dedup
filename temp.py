import cyrtranslit
from preprocessing.textsim import get_sim_features, compare
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("russian", ignore_stopwords=True)


q1 = 'estel always on line mousse normal hold 336.0'
d1 = 'мусс для волос нормальная фиксация on line estel'

q2 = cyrtranslit.to_cyrillic(q1, 'ru')
d2 = cyrtranslit.to_cyrillic(d1, 'ru')

compare(q1, d1, q2, d2)

stemmer.stem('персик')
