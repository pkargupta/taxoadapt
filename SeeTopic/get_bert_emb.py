import numpy as np
import argparse
import torch
import os
import pickle as pk
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# map each term in text to word_id

def get_vocab_idx(split_text: str, tok_lens):

	vocab_idx = {}
	start = 0

	for w in split_text:
		# print(w, start, start + len(tok_lens[w]))
		if w not in vocab_idx:
			vocab_idx[w] = []

		vocab_idx[w].extend(np.arange(start, start + len(tok_lens[w])))

		start += len(tok_lens[w])

	return vocab_idx

def get_hidden_states(encoded, data_idx, model, layers, static_emb):
	"""Push input IDs through model. Stack and sum `layers` (last four by default).
	Select only those subword token outputs that belong to our word of interest
	and average them."""
	with torch.no_grad():
		output = model(**encoded)

	# Get all hidden states
	states = output.hidden_states
	# Stack and sum all requested layers
	output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

	# Only select the tokens that constitute the requested word

	for w in data_idx:
		static_emb[w] += output[data_idx[w]].sum(dim=0).cpu().numpy()

def chunkify(text, token_lens, length=512):
	chunks = [[]]
	text_chunks = []
	split_text = text.split()
	count = 0
	for word in split_text:
		new_count = count + len(token_lens[word]) + 2 # 2 for [CLS] and [SEP]
		if new_count > length:
			chunks.append([word])
			count = len(token_lens[word])
		else:
			chunks[len(chunks) - 1].append(word)
			count = new_count
	
	return chunks


device = torch.device("cuda")

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='scidocs')
parser.add_argument('--model', default='bert')
parser.add_argument('--dim', default=768)
parser.add_argument('--length', default=512)
parser.add_argument('--batch_size', default=128)
args = parser.parse_args()

if args.model == 'bert_cls_ft':
	bert_model = '/home/pk36/Comparative-Summarization/bert_cls_ft/checkpoint-1080/'
elif args.model == 'bert_nlp_ft':
	bert_model = '/home/pk36/Comparative-Summarization/bert_nlp_ft/checkpoint-6252/'
elif args.model == 'bert_full_ft':
	bert_model = '/home/pk36/Comparative-Summarization/bert_full_ft/checkpoint-8346/'
elif args.model == 'bert':
	bert_model = 'bert-base-uncased'
elif args.model == 'deberta':
	bert_model = 'google/electra-base-discriminator'
else:
	bert_model = 'biobert-v1.1/'

corpus_file = f'{args.dataset}.txt'
bert_file = f'embedding_{args.model}.txt'

tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

print("####### CONSTRUCTING AND TOKENIZING VOCAB #######")

# get total frequency:
cnt = defaultdict(int)
token_lens = {}
static_emb = {}

with open(os.path.join(args.dataset, corpus_file)) as fin:
	for line in fin:
		data = line.strip().split()
		for word in data:
			cnt[word] += 1
			if word not in token_lens:
				token_lens[word] = tokenizer.tokenize(word.replace("_", " "))
				static_emb[word] = np.zeros((args.dim, ))


tokenized_docs = []
tokenized_sents = []
layers = [-4, -3, -2, -1]

print("####### COMPUTING STATIC EMBEDDINGS #######")

if os.path.exists(os.path.join(args.dataset, 'static_emb.pk')):
	with open(os.path.join(args.dataset, 'static_emb.pk'), "rb") as f:
		saved_emb = pk.load(f)
		static_emb = saved_emb["static_emb"]
		token_lens = saved_emb["token_lens"]
		tokenized_sents = saved_emb["tokenized_sents"]
		tokenized_docs = saved_emb["tokenized_docs"]
else:

	with open(os.path.join(args.dataset, corpus_file)) as fin:
		lines = [l.strip() for l in fin]
		for doc_id, doc in tqdm(enumerate(lines), total=len(lines)):
			chunks = chunkify(doc, token_lens, args.length)

			tokenized_docs.append([sent for sent in doc.split(" . ")])
			tokenized_sents.append([sent.split() for sent in tokenized_docs[doc_id]])
			for chunk_id, chunk in enumerate(chunks):
				# get indices and frequencies for each term in data
				data_idx = get_vocab_idx(chunk, token_lens)

				# compute contextualized word embeddings
				encoded_data = tokenizer.encode_plus(" ".join(chunk).replace("_", " "), return_tensors="pt").to(device)

				get_hidden_states(encoded_data, data_idx, model, layers, static_emb)

	static_emb = {w:w_emb/cnt[w] for w, w_emb in static_emb.items()}

	with open(os.path.join(args.dataset, 'static_emb.pk'), "wb") as f:
		pk.dump({"static_emb": static_emb, "token_lens": token_lens, "tokenized_sents": tokenized_sents, "tokenized_docs": tokenized_docs}, f)

print("####### COMPUTING RAW WORD EMBEDDINGS #######")


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

vocabulary = list(vocabulary)

with open(os.path.join(args.dataset, bert_file), 'w') as f:
	f.write(f'{len(vocabulary)} {args.dim}\n')

	for i in tqdm(range(0, len(vocabulary), args.batch_size)):
		batch_vocab = vocabulary[i:i+args.batch_size]
		batch_texts = [word.replace('_', ' ') for word in batch_vocab]
		encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
		input_ids = encoded_inputs['input_ids'].to(device)
		attention_mask = encoded_inputs['attention_mask'].to(device)

		with torch.no_grad():
			outputs = model(input_ids, attention_mask=attention_mask)

		hidden_states = outputs.hidden_states[-1]
		mean_embeddings = torch.mean(hidden_states, dim=1).cpu()

		for word, emb in zip(batch_vocab, mean_embeddings):
			f.write(f'{word} ' + ' '.join([str(x.item()) for x in emb]) + '\n')
		

		# for word in tqdm(vocabulary):
		# 	text = word.replace('_', ' ')
		# 	input_ids = torch.tensor(tokenizer.encode(text, max_length=256, truncation=True)).unsqueeze(0).to(device)
		# 	outputs = model(input_ids)
		# 	hidden_states = outputs[2][-1][0]

		# 	# take the average between the static embedding (if it exists) and the raw bert embedding
		# 	# if word in static_emb:
		# 	# 	emb = (torch.mean(hidden_states, dim=0).cpu() + static_emb[word].cpu())/2
		# 	# else:
		# 	emb = torch.mean(hidden_states, dim=0).cpu()

		# 	f.write(f'{word} ' + ' '.join([str(x.item()) for x in emb])+'\n')
