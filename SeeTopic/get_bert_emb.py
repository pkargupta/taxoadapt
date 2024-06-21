import numpy as np
import argparse
import torch
import os
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='scidocs')
parser.add_argument('--model', default='bert')
args = parser.parse_args()

if args.model == 'bert_cls_ft':
	bert_model = '/home/pk36/Comparative-Summarization/bert_cls_ft/checkpoint-1080/'
elif args.model == 'bert_nlp_ft':
	bert_model = '/home/pk36/Comparative-Summarization/bert_nlp_ft/checkpoint-6252/'
elif args.model == 'bert_full_ft':
	bert_model = '/home/pk36/Comparative-Summarization/bert_full_ft/checkpoint-8346/'
elif args.model == 'bert':
	bert_model = 'bert-base-uncased'
else:
	bert_model = 'biobert-v1.1/'

corpus_file = f'{args.dataset}.txt'
bert_file = f'embedding_{args.model}.txt'

tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

cnt = defaultdict(int)
with open(os.path.join(args.dataset, corpus_file)) as fin:
	for line in fin:
		data = line.strip().split()
		for word in data:
			cnt[word] += 1

min_count = 3
vocabulary = set()
for word in cnt:
	if cnt[word] >= min_count and word.replace('_', ' ').strip() != '':
		vocabulary.add(word)

with open(os.path.join(args.dataset, 'keywords_0.txt')) as fin, open(os.path.join(args.dataset, 'oov.txt'), 'w') as fout:
	for line in fin:
		word = line.strip().split(':')[1]
		# NEW: factor in seed words
		if ',' in word:
			words = word.split(',')
			for w in words:
				if w not in vocabulary:
					fout.write(w+'\n')
					vocabulary.add(w)
		else:
			if word not in vocabulary:
				fout.write(word+'\n')
				vocabulary.add(word)

with torch.no_grad():
	with open(os.path.join(args.dataset, bert_file), 'w') as f:
		f.write(f'{len(vocabulary)} 768\n')
		for word in tqdm(vocabulary):
			text = word.replace('_', ' ')
			input_ids = torch.tensor(tokenizer.encode(text, max_length=256, truncation=True)).unsqueeze(0).to(device)
			outputs = model(input_ids)
			hidden_states = outputs[2][-1][0]
			emb = torch.mean(hidden_states, dim=0).cpu()

			f.write(f'{word} '+' '.join([str(x.item()) for x in emb])+'\n')
