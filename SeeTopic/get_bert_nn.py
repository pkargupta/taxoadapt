import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='scidocs')
parser.add_argument('--model', default='bert')
parser.add_argument('--num_iter', default=0, type=int)
args = parser.parse_args()

dataset = args.dataset
model = args.model
num_iter = args.num_iter
topk = 3*num_iter+1

topics = []
seed_words = []
# NEW: ADDED COMPATABILITY WITH SEED WORDS --> make sure that the new topic words selected for cate are chosen using seed words
with open(f'{dataset}/keywords_0.txt') as fin:
	for line in fin:
		data = line.strip().split(':')[1].split(',')
		topics.append(data)
		seed_words.append(data)

if num_iter > 0:
	with open(f'{dataset}/keywords_{num_iter}.txt') as fin:
		for idx, line in enumerate(fin):
			data = line.strip().split(':')[1].split(',')
			topics[idx].extend(data[1:topk])

# TODO: add in static word embeddings here??
word2emb = {}
with open(f'{dataset}/embedding_{model}.txt') as fin:
	for line in fin:
		data = line.strip().split()
		if len(data) != 769:
			continue
		word = data[0]
		emb = np.array([float(x) for x in data[1:]])
		emb = emb / np.linalg.norm(emb)
		word2emb[word] = emb

oov = set()
with open(f'{dataset}/oov.txt') as fin:
	for line in fin:
		data = line.strip()
		oov.add(data)

if num_iter == 0:
	out_file = f'{dataset}/keywords_1.txt'
else:
	num_iter += 1
	out_file = f'{dataset}/keywords_{model}_{num_iter}.txt'

with open(out_file, 'w') as fout:
	for idx, topic in enumerate(topics):
		word2score = defaultdict(float)
		for word in word2emb:
			if word in oov:
				continue
			for term in topic:
				word2score[word] += np.dot(word2emb[word], word2emb[term])
		score_sorted = sorted(word2score.items(), key=lambda x: x[1], reverse=True)[:100]
		# NEW: ADDED IN VOCABULARY SEED WORDS
		new_topic = [topic[0]] + [i for i in seed_words[idx][1:] if not (i in oov)] + [x[0] for x in score_sorted]
		new_topic = list(dict.fromkeys(new_topic))
		fout.write(f'{idx}:'+','.join(new_topic)+'\n')
