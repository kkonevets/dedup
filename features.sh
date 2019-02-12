# run from ubuntu@10.72.102.67

RED='\033[0;31m'

cd dedup
export PYTHONPATH="$PYTHONPATH:$PWD"
echo $PWD

printf "${RED}dataset\n"
~/miniconda3/bin/python preprocessing/dataset.py \
--data_dir=../data/dedup/phase1/ \
--build_features \
--build_tfidf \
--tfidf