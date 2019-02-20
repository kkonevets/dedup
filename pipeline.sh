# exit when any command fails
set -e

RED='\033[0;31m'
NC='\033[0m'

dst="ubuntu@10.72.102.67"

export PYTHONPATH="$PYTHONPATH:$PWD"

# printf "${RED}sampling\n${NC}"
# python3 preprocessing/sampling.py \
# --data_dir=../data/dedup/ \
# --nrows=50
# --for_test

printf "${RED}corpus\n${NC}"
python3 preprocessing/corpus.py \
--data_dir=../data/dedup/ \
--build_tfidf

printf "${RED}coping to ${NC}$dst\n"
rsync -amvzP ../data/dedup/* $dst:/home/ubuntu/data/dedup

ssh $dst 'bash -s' < features.sh
