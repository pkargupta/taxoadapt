# main.py

import json
import argparse
import os
import numpy as np
from utils import get_json_files
from analysis import analyze_json
from evaluation import get_node_wise_uniqueness_equivalent

DIMS = ['task', 'methodology', 'datasets', 'evaluation methods', 'real-world domains']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str)
#     parser.add_argument("--output_path", type=str)
    parser.add_argument("--corpus_path", type=str, default='taxoadapt/datasets/emnlp_2024/internal.txt')
    parser.add_argument("--model_judge_name", type=str, default="gpt-4o", help="Name of the LLM model")
    args = parser.parse_args()
    
    id2paper = {}
    with open(args.corpus_path) as f:
        for i, line in enumerate(f):
            id2paper[i] = json.loads(line.strip())
    
    final_results = {}
    matched = set()
    for dim in DIMS:
        
        print(f'Evaluating Dimension: {dim}')

        # Load JSON file
        eval_json = get_json_files(os.path.join(args.eval_path, f'{dim}.json'.replace(' ', '_')))
        root, taxonomy, paths, levels, nodes = analyze_json(eval_json)

        # Evaluate taxonomy uniqueness
        node_wise_uniqueness = get_node_wise_uniqueness_equivalent(root, nodes, taxonomy)
        valid_uniqueness_scores = [item['score'] for item in node_wise_uniqueness if item['score'] != -1]
        avg_uniqueness = (sum(valid_uniqueness_scores) / len(valid_uniqueness_scores)) if valid_uniqueness_scores else 0
        print(avg_uniqueness)
        
        final_results[dim] = {
            "avg_uniqueness_score": avg_uniqueness,
            "node_wise_uniqueness": node_wise_uniqueness,
        }
        with open(os.path.join(args.eval_path, 'eval_unique_new.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
    all_result = {
        "avg_uniqueness_score": np.mean([final_results[dim]['avg_uniqueness_score'] for dim in DIMS]),
    }
    final_results['all'] = all_result
    
    with open(os.path.join(args.eval_path, 'eval_unique_new.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(all_result)

if __name__ == "__main__":
    main()
