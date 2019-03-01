from absl import app
from gensim.models import FastText
from pymongo import MongoClient
import tools
from itertools import chain
from tokenizer import tokenize

FLAGS = tools.FLAGS


def main(argv):
    client = MongoClient(FLAGS.mongo_host)
    mdb = client[FLAGS.release_db]
    mets = mdb.etalons.find({},
                            projection=['name', 'description', 'synonyms', 'mistypes'])
    mtotal = mdb.etalons.count_documents({})

    samples = tools.load_samples('../data/dedup/samples.npz')
    samples = samples[samples['train'] == 0]
    test_ids = list(samples['qid'].unique())

    db = client[FLAGS.feed_db]
    cond = {'_id': {'$nin': test_ids}}
    ets = db.etalons.find(cond,
                          projection=['name', 'description', 'synonyms', 'mistypes'])
    total = db.etalons.count_documents(cond)

    def iter_sents(ets, total):
        for et in tools.tqdm(ets, total=total):
            name = tokenize(et['name'])
            yield tools.replace_num(name.split())

            desc = et.get('description')
            if desc:
                desc = tokenize(desc)
                yield tools.replace_num(desc.split())

            for syn in et.get('synonyms', []) + et.get('mistypes', []):
                sname = tokenize(syn['name'])
                yield tools.replace_num(sname.split())

    sents = list(iter_sents(chain(ets, mets), total+mtotal))

    model = FastText(size=100, window=5, min_count=5,
                     workers=-1)  # instantiate
    model.build_vocab(sentences=sents)
    model.train(sentences=sents, total_examples=len(
        sents), epochs=30)  # train

    model.save('../data/dedup/ftext.model')

    model.wv.most_similar(['мыло'])
    # [('пломбирс', 0.7868414521217346),
    # ('пломбирь', 0.7198898792266846),
    # ('пломбирное', 0.7051539421081543),
    # ('пломбирная', 0.7040817141532898),
    # ('пломбира', 0.6881129741668701),
    # ('пломбироешка', 0.6706045866012573),
    # ('пломбиретто', 0.6349174976348877),
    # ('пломбирныи', 0.6323882937431335),
    # ('пломбиром', 0.6218162178993225),
    # ('пломбис', 0.5896835923194885)]


if __name__ == '__main__':
    import __main__

    if hasattr(__main__, '__file__'):
        app.run(main)
    else:
        tools.sys.argv += []
        FLAGS(tools.sys.argv)
