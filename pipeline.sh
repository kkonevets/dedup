RED='\033[0;31m'
NC='\033[0m'

dst="ubuntu@10.72.102.67"

printf "${RED}sampling\n"
python3 preprocessing/sampling.py \
--data_dir=../data/dedup/phase1/ \
--nrows=20 \
--nchoices=20 

printf "${RED}corpus\n"
python3 preprocessing/corpus.py \
--data_dir=../data/dedup/phase1 \
--build_tfidf

printf "${RED}coping to ${NC}$dst\n"
scp ../data/dedup/tfidf_model.pkl $dst:/home/ubuntu/data/dedup
scp -r -C ../data/dedup/phase1/* $dst:/home/ubuntu/data/dedup/phase1

ssh $dst 'bash -s' < features.sh
