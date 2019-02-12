printf "sampling\n"
python3 preprocessing/sampling.py \
--data_dir=../data/dedup/phase1/ \
--nrows=20 \
--nchoices=20 

printf "corpus\n"
python3 preprocessing/corpus.py \
--data_dir=../data/dedup/phase1 \
--build_tfidf

printf "dataset\n"
python3 preprocessing/dataset.py \
--data_dir=../data/dedup/phase1/ \
--build_features \
--build_tfidf \
--tfidf
