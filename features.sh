# run from ubuntu@10.72.102.67

RED='\033[0;31m'
NC='\033[0m'

cd dedup
export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PWD

printf "${RED}dataset\n${NC}"
~/miniconda3/bin/python preprocessing/dataset.py \
--data_dir=../data/dedup/ \
--build_features \
--build_tfidf \
--tfidf