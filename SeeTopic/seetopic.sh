#!/bin/bash
DATASET=$1
ITER=$2
RHO=0.1
MODEL=$3

# if [ "$DATASET" == "scidocs" ]; then
# 	MODEL=biobert
# else
# 	MODEL=bert
# fi

green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}===Get PLM Embeddings===${reset}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python get_bert_emb.py --dataset $DATASET --model $MODEL

echo "${green}===Iter 0: PLM Module===${reset}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python get_bert_nn.py --dataset $DATASET --model $MODEL --num_iter 0

for i in $(seq $ITER); do
	echo "${green}===Iter $i: PLM Module===${reset}"
	CUDA_VISIBLE_DEVICES=0,1,2,3,4 python get_bert_nn.py --dataset $DATASET --model $MODEL --num_iter $i

	echo "${green}===Iter $i: Local Module===${reset}"
	cd seed_guided_emb/
	./run_local.sh $DATASET $i
	cd ../

	echo "${green}===Iter $i: Ensemble===${reset}"
	CUDA_VISIBLE_DEVICES=0,1,2,3,4 python ensemble.py --dataset $DATASET --model $MODEL --num_iter $(($i+1)) --rho $RHO
done

cp $DATASET/keywords_$(($ITER+1)).txt $DATASET/keywords_seetopic.txt