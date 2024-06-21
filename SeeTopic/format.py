import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='scidocs')
parser.add_argument('--model', default='bert')
args = parser.parse_args()

dataset = args.dataset
model = args.model

topk = 10
with open(f'{dataset}/keywords_{model}.txt') as fin, open(f'{dataset}/keywords_{model}.csv', 'w') as fout:
	for line in fin:
		data = line.strip().split(':')[1].split(',')[:topk+1]
		fout.write('\t'.join(data)+'\n')