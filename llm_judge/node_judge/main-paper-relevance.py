# main.py

import json
import argparse
import os
import numpy as np
from utils import get_json_files
from analysis import analyze_json
from evaluation import get_node_wise_paper_relevance

DIMS = ['task', 'methodology', 'datasets', 'evaluation methods', 'real-world domains']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_path", type=str)
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

        # Evaluate node segment quality
        node_wise_segment_quality = get_node_wise_paper_relevance(root, nodes, id2paper, 30)
        paper_relevance_scores = []
        for node_result in node_wise_segment_quality:
            paper_relevance_scores.append(1 if len(node_result['relevant']) >= 30 else 0)
#             matched.update(node_result['relevant'])
        avg_paper_relevance = (sum(paper_relevance_scores) / len(paper_relevance_scores))
        print(avg_paper_relevance)
        
        final_results[dim] = {
            "avg_paper_relevance": avg_paper_relevance,
            "node_wise_paper_relevance": node_wise_segment_quality
        }
        with open(os.path.join(args.eval_path, 'eval_v2.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
    all_result = {
        "avg_paper_relevance": np.mean([final_results[dim]['avg_paper_relevance'] for dim in DIMS]),
#         "coverage": len(matched) / len(id2paper)
    }
    final_results['all'] = all_result
    
    with open(os.path.join(args.eval_path, 'eval_v2.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(all_result)

if __name__ == "__main__":
    main()
