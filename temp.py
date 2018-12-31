from nltk.stem.snowball import SnowballStemmer
from transliterate import translit, get_available_language_codes

stemmer = SnowballStemmer("russian")

words = ['индейка', 'магнитный']
for w in words:
    print(stemmer.stem(w))


text = 'coca'
print(translit(text, 'ru'))
