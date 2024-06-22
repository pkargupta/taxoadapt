dataset=$1
iter=$2
text_file=../${dataset}/${dataset}.txt
topic_file=keywords.txt
topic=$(echo ${topic_file} | cut -d'.' -f 1)

cd src
make cate
cd ../

mkdir -p ${dataset}_${iter}/

python3 prepare.py --dataset ${dataset} --num_iter ${iter}

./src/cate -train ${text_file} -topic-name ${dataset}_${iter}/${topic_file} \
	-res ${dataset}_${iter}/res_cate.txt -k 10 -expand 1 \
	-word-emb ${dataset}_${iter}/emb_cate_w.txt -topic-emb ${dataset}_${iter}/emb_cate_t.txt \
	-size 768 -window 5 -negative 5 -sample 1e-3 -min-count 3 \
	-threads 20 -binary 0 -iter 10 -pretrain 2

python3 get_local_nn.py --dataset ${dataset} --num_iter ${iter}

