import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='scidocs')
parser.add_argument('--num_iter', default=0, type=int)
args = parser.parse_args()

dataset = args.dataset
num_iter = args.num_iter
topk = 3*num_iter+1

topics = []
with open(f'../{dataset}/keywords_{num_iter}.txt') as fin:
	for line in fin:
		data = line.strip().split(':')[1].split(',')
		topics.append(data[:topk])

word2emb = {}
with open(f'{dataset}_{num_iter}/emb_cate_w.txt') as fin:
	for line in fin:
		data = line.strip().split()
		if len(data) != 769:
			continue
		word = data[0]
		emb = np.array([float(x) for x in data[1:]])
		emb = emb / np.linalg.norm(emb)
		word2emb[word] = emb

num_iter += 1
out_file = f'../{dataset}/keywords_local_{num_iter}.txt'

with open(out_file, 'w') as fout:
	for idx, topic in enumerate(topics):
		word2score = defaultdict(float)
		for word in word2emb:
			for term in topic[1:]:
				word2score[word] += np.dot(word2emb[word], word2emb[term])
		score_sorted = sorted(word2score.items(), key=lambda x: x[1], reverse=True)[:100]
		new_topic = [topic[0]]+[x[0] for x in score_sorted]
		fout.write(f'{idx}:'+','.join(new_topic)+'\n')
