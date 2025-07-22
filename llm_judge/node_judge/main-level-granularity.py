# main.py

import json
import argparse
import os
import numpy as np
from utils import get_json_files
from analysis import analyze_json
from evaluation import (
    get_level_granularity_new,
)

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

#         # Evaluate dimension alignment
#         dimension_alignment = get_dimension_alignment(root, dim, nodes)
#         valid_alignment_scores = [item['score'] for item in dimension_alignment if item['score'] != -1]
#         avg_alignment = (sum(valid_alignment_scores) / len(valid_alignment_scores)) if valid_alignment_scores else 0
#         print(avg_alignment)

#         # Evaluate path granularity
#         path_wise_granularity = get_path_granularity(root, paths)
#         valid_granularity_scores = [item['score'] for item in path_wise_granularity if item['score'] != -1]
#         avg_granularity = (sum(valid_granularity_scores) / len(valid_granularity_scores)) if valid_granularity_scores else 0
#         print(avg_granularity)

        # Evaluate level granularity
        level_wise_granularity = get_level_granularity_new(root, levels)
        valid_level_scores = [level['score'] for level in level_wise_granularity if level['score'] != -1]
        avg_level_granularity = (sum(valid_level_scores) / len(valid_level_scores)) if valid_level_scores else 0
        print(avg_level_granularity)

#         # Evaluate taxonomy uniqueness
#         node_wise_uniqueness = get_node_wise_uniqueness(root, nodes, taxonomy)
#         valid_uniqueness_scores = [item['score'] for item in node_wise_uniqueness if item['score'] != -1]
#         avg_uniqueness = (sum(valid_uniqueness_scores) / len(valid_uniqueness_scores)) if valid_uniqueness_scores else 0
#         print(avg_uniqueness)

#         # Evaluate node segment quality
#         node_wise_segment_quality = get_node_wise_segment_quality(root, nodes, id2paper)
#         paper_relevance_scores = []
#         for node_result in node_wise_segment_quality:
#             paper_relevance_scores.append((len(node_result['relevant']) / len(node_result['valid'])) if node_result['valid'] else 0)
#             matched.update(node_result['relevant'])
#         avg_paper_relevance = (sum(paper_relevance_scores) / len(paper_relevance_scores))
#         print(avg_paper_relevance)
# #         node_wise_segment_quality = get_node_wise_segment_quality_per_paper(root, nodes[:5], id2paper)
# #         paper_relevance_scores = []
# #         for node_result in node_wise_segment_quality:
# #             valid_scores = [s for s in node_result['scores'] if s != -1]
# #             paper_relevance_scores.append((sum(valid_scores) / len(valid_scores)) if valid_scores else 0)
# #             matched.update([paper_id for paper_id, s in zip(node_result['node']['indices'], node_result['scores']) if s == 1])
# #         avg_paper_relevance = (sum(paper_relevance_scores) / len(paper_relevance_scores))
        
        final_results[dim] = {
#             "avg_alignment_score": avg_alignment,
#             "node_wise_dimension_alignment": dimension_alignment,
#             "avg_granularity_score": avg_granularity,
#             "path_wise_granularity": path_wise_granularity,
            "avg_level_granularity_score": avg_level_granularity,
            "level_wise_granularity": level_wise_granularity,
#             "avg_uniqueness_score": avg_uniqueness,
#             "node_wise_uniqueness": node_wise_uniqueness,
#             "avg_paper_relevance": avg_paper_relevance,
#             "node_wise_paper_relevance": node_wise_segment_quality
        }
        with open(os.path.join(args.eval_path, 'eval_level_new1.json'), 'w') as f:
            json.dump(final_results, f, indent=4)
    all_result = {
#         "avg_alignment_score": np.mean([final_results[dim]['avg_alignment_score'] for dim in DIMS]),
#         "avg_granularity_score": np.mean([final_results[dim]['avg_granularity_score'] for dim in DIMS]),
        "avg_level_granularity_score": np.mean([final_results[dim]['avg_level_granularity_score'] for dim in DIMS]),
#         "avg_uniqueness_score": np.mean([final_results[dim]['avg_uniqueness_score'] for dim in DIMS]),
#         "avg_paper_relevance": np.mean([final_results[dim]['avg_paper_relevance'] for dim in DIMS]),
#         "coverage": len(matched) / len(id2paper)
    }
    final_results['all'] = all_result
    
    with open(os.path.join(args.eval_path, 'eval_level_new1.json'), 'w') as f:
        json.dump(final_results, f, indent=4)
        
    print(all_result)

if __name__ == "__main__":
    main()
