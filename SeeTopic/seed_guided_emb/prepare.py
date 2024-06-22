import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='scidocs')
parser.add_argument('--num_iter', default=1, type=int)
args = parser.parse_args()

dataset = args.dataset
num_iter = args.num_iter
topk = 3*num_iter+1

with open(f'../{dataset}/keywords_{num_iter}.txt') as fin, open(f'{dataset}_{num_iter}/keywords.txt', 'w') as fout:
	for line in fin:
		data = line.strip().split(':')[1].split(',')
		seeds = data[1:topk]
		fout.write(' '.join(seeds)+'\n')