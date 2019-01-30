java -jar RankLib-2.11.jar -train train_letor.txt -tvs 0.8 -test test_letor.txt -ranker 6 -metric2t P@1 -metric2T P@1 -save lmart.txt
java -jar RankLib-2.11.jar -train train_letor.txt -tvs 0.8 -test test_letor.txt -ranker 7 -metric2T NDCG@1 -save lnet.txt

java -jar RankLib-2.11.jar -load lmart.txt -rank test_letor.txt -score myScoreFile.txt

java -jar RankLib-2.11.jar -train train_letor.txt -validate vali_letor.txt -test test_letor.txt -ranker 6 -metric2t NDCG@1 -metric2T NDCG@1 -save lmart.txt
