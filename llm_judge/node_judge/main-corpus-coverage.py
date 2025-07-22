# main.py

import json
import argparse
import os
import numpy as np
from utils import get_json_files
from analysis import analyze_json
from evaluation import get_node_wise_paper_relevance_all
from collections import defaultdict
from tqdm import tqdm

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
    for dim in tqdm(DIMS):
        
        print(f'Evaluating Dimension: {dim}')

        # Load JSON file
        eval_json = get_json_files(os.path.join(args.eval_path, f'{dim}.json'.replace(' ', '_')))
        root, taxonomy, paths, levels, nodes = analyze_json(eval_json)


        # Evaluate node segment quality
        node_wise_paper_relevance = get_node_wise_paper_relevance_all(root, nodes, id2paper)
        paper_relevance_scores = []
        node2paper = {}
        for node_result in node_wise_paper_relevance:
            paper_relevance_scores.append(1 if len(node_result['relevant']) >= len(id2paper)*0.05 else 0)
            node2paper[node_result['node']] = node_result['relevant']
        avg_paper_relevance = (sum(paper_relevance_scores) / len(paper_relevance_scores))
        print(avg_paper_relevance)
        
        level_coverage_scores = []
        for i, level in enumerate(levels):
            if level['parent'] == root:
                parent_papers = set(id2paper.keys())
            else:
                parent_papers = node2paper[level['parent']]
            children_papers = set()
            for child in level['siblings']:
                if child == root:
                    parent_papers = set(id2paper.keys())
                else:
                    children_papers.update(node2paper[child])
            level_coverage_scores.append(len(children_papers.intersection(parent_papers)) / len(parent_papers))
        avg_coverage_score = sum(level_coverage_scores)/len(level_coverage_scores)
        print(avg_coverage_score)
        
        final_results[dim] = {
            "avg_paper_relevance": avg_paper_relevance,
            "avg_coverage_score": avg_coverage_score,
            "node_wise_paper_relevance": node_wise_paper_relevance,
        }
        with open(os.path.join(args.eval_path, 'eval_cover_final.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
    all_result = {
        "avg_paper_relevance": np.mean([final_results[dim]['avg_paper_relevance'] for dim in DIMS]),
        "avg_coverage_score": np.mean([final_results[dim]['avg_coverage_score'] for dim in DIMS])
    }
    final_results['all'] = all_result
    
    with open(os.path.join(args.eval_path, 'eval_cover_final.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(all_result)

if __name__ == "__main__":
    main()
