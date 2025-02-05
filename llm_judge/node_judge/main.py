# main.py

import json
import argparse
from llm_judge.node_judge.utils import get_json_files
from llm_judge.node_judge.analysis import analyze_json
from llm_judge.node_judge.evaluation import (
    get_path_relevance,
    get_path_granularity,
    get_level_granularity,
    get_taxonomy_wise_uniqueness,
    get_node_wise_segment_quality,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json_path", type=str, default="eval/example/hierarchy.json")
    parser.add_argument("--output_path", type=str, default="eval/example/node_llm_judge.json")
    parser.add_argument("--model_judge_name", type=str, default="gpt-4o", help="Name of the LLM model")
    args = parser.parse_args()

    # Load JSON file
    eval_json = get_json_files(args.eval_json_path)
    claim, taxonomy, paths, levels, nodes = analyze_json(eval_json)
    
    # Evaluate path relevance
    path_wise_relevance = get_path_relevance(claim, paths)
    valid_relevance_scores = [item['score'] for item in path_wise_relevance if item['score'] != -1]
    avg_relevance = (sum(valid_relevance_scores) / len(valid_relevance_scores)) if valid_relevance_scores else 0
    
    # Evaluate path granularity
    path_wise_granularity = get_path_granularity(claim, paths)
    valid_granularity_scores = [item['score'] for item in path_wise_granularity if item['score'] != -1]
    avg_granularity = (sum(valid_granularity_scores) / len(valid_granularity_scores)) if valid_granularity_scores else 0
        
    # Evaluate level granularity
    level_wise_granularity = get_level_granularity(claim, levels)
    valid_level_scores = [level['score'] for level in level_wise_granularity if level['score'] != -1]
    avg_level_granularity = (sum(valid_level_scores) / len(valid_level_scores)) if valid_level_scores else 0
    
    # Evaluate taxonomy uniqueness
    taxonomy_wise_uniqueness = get_taxonomy_wise_uniqueness(claim, taxonomy)
    
    # Evaluate node segment quality
    node_wise_segment_quality = get_node_wise_segment_quality(claim, nodes)
    valid_segment_scores = [node['score'] for node in node_wise_segment_quality if node['score'] != -1]
    avg_segment_quality = (sum(valid_segment_scores) / len(valid_segment_scores)) if valid_segment_scores else 0
        
    final_results = {
        "avg_relevance_score": avg_relevance,
        "avg_granularity_score": avg_granularity,
        "avg_level_granularity_scpre": avg_level_granularity,
        "taxonomy_wise_uniqueness_score": taxonomy_wise_uniqueness['score'],
        "avg_segment_quality_scpre": avg_segment_quality,
        "path_wise_relevance": path_wise_relevance,
        "path_wise_granularity": path_wise_granularity,
        "level_wise_granularity": level_wise_granularity,
        "taxonomy_wise_uniqueness": taxonomy_wise_uniqueness,
        "node_wise_segment_quality": node_wise_segment_quality
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()
