import argparse
import os
import csv
import string
import numpy as np
import re
from tqdm import tqdm
import json
from nltk import word_tokenize

# after AutoPhrase
def phrase_process():
	f = open(os.path.join('AutoPhrase', 'models', args.dataset, 'segmentation.txt'))
	g = open(args.out_file, 'w')
	for line in tqdm(f):
		doc = ''
		temp = line.split('<phrase>')
		for seg in temp:
			temp2 = seg.split('</phrase>')
			if len(temp2) > 1:
				doc += ('_').join(temp2[0].split(' ')) + temp2[1]
			else:
				doc += temp2[0]
		doc = re.sub(r'(\w)-(\w)', r'\1_\2', doc)
		g.write(doc.strip()+'\n')
	print("Phrase segmented corpus written to {}".format(args.out_file))
	return 

# before AutoPhrase
def preprocess():
	printable = set(string.printable)
	f = open(os.path.join(args.dataset, args.in_file))
	docs = f.readlines()
	f_out = open(args.out_file, 'w')
	for doc in tqdm(docs):
		if doc[0] == "{":
			doc_json = json.loads(doc.strip())
			if 'external' in args.in_file:
				doc = f'paper_title : {doc_json["Title"]} ; paper_abstract : {doc_json["Abstract"]}'
			else:
				doc = f'paper_title : {doc_json["Title"]} ; paper_abstract : {doc_json["Abstract"]} ; paper_content : {doc_json["Content"]}'

		if '. References ' in doc:
			doc = doc.split('. References ')[0]
		doc = ''.join(filter(lambda x: x in printable, doc)).replace('\r', '')
		doc = re.sub(r'\/uni\w{4,8}', "", doc)
		f_out.write(' '.join([w.lower().replace('-', '_') for w in word_tokenize(doc.strip())]) + '\n')
	return 


if __name__=="__main__":
	
	parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--mode', type=int)
	parser.add_argument('--dataset', default='NEW')
	parser.add_argument('--in_file', default='text.txt')
	parser.add_argument('--out_file', default='./AutoPhrase/data/text.txt')
	args = parser.parse_args()

	if args.mode == 0:
		preprocess()
	else:
		phrase_process()
	
				