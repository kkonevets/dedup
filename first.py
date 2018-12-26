import json
import urllib
from urllib.parse import quote
from pydev import updating


def query_solr(text, rows=1, exclude=[]):
    q = 'http://c:8983/solr/nom_core/select?df=my_text_ru&fl=categoryId&fq' \
        '=-categoryId:(%s)&q=%s&rows=%d' % (quote(' '.join(exclude)),
                                            quote(text), rows)
    r = urllib.request.urlopen(q).read()
    docs = json.loads(r)['response']['docs']
    if not len(docs):
        return None
    cid = docs[0]['categoryId']
    return cid
