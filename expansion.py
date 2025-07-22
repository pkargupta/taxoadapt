import json
import argparse
from pydantic import BaseModel, StringConstraints
from typing_extensions import Annotated
from typing import Dict
from collections import Counter

from taxonomy import Node
from utils import clean_json_string
from model_definitions import constructPrompt, promptLLM
from prompts import width_system_instruction, width_main_prompt, WidthExpansionSchema, width_cluster_system_instruction, width_cluster_main_prompt, WidthClusterListSchema
from prompts import depth_system_instruction, depth_main_prompt, DepthExpansionSchema, depth_cluster_system_instruction, depth_cluster_main_prompt, DepthClusterListSchema

######################## WIDTH EXPANSION ########################

def expandNodeWidth(args, node, id2node, label2node):
    unlabeled_papers = {}
    for idx, p in node.papers.items():
        unlabeled = True
        for c in node.children.values():
            if idx in c.papers:
                unlabeled = False
                break
        if unlabeled:
            unlabeled_papers[idx] = p
    
    node_ancestors = node.get_ancestors()
    if node_ancestors is None:
        ancestors = "None"
    else:
        node_ancestors.reverse()
        ancestors = " -> ".join([ancestor.label for ancestor in node_ancestors])

    print(f'node {node.label} ({node.dimension}) has {len(unlabeled_papers)} unlabeled papers!')

    if len(unlabeled_papers) <= args.max_density:
        return [] 
    
    exp_prompts = [constructPrompt(args, width_system_instruction, width_main_prompt(paper, node, ancestors)) for paper in unlabeled_papers.values()]
    exp_outputs = promptLLM(args=args, prompts=exp_prompts, schema=WidthExpansionSchema, max_new_tokens=300, json_mode=True, temperature=0.6, top_p=0.99)
    exp_outputs = [json.loads(clean_json_string(c))['new_subtopic_label'].replace(' ', '_').lower() 
                   if "```" in c else json.loads(c.strip())['new_subtopic_label'].replace(' ', '_').lower() 
                   for c in exp_outputs]

    exp_outputs = [w for w in exp_outputs if w + f"_{node.dimension}" not in label2node]
    if len(exp_outputs) == 0:
        return []
    freq_options = dict(Counter(exp_outputs))

    print(freq_options)

    all_node_labels = ", ".join(list(label2node.keys()))

    # FILTERING OF EXPANSION OUTPUTS
    args.llm = 'gpt'
    clustered_prompt = [constructPrompt(args, width_cluster_system_instruction, width_cluster_main_prompt(freq_options, node, ancestors, all_node_labels))]
    success = False
    attempts = 0
    while (not success) and (attempts < 5):
        try:
            cluster_topics = promptLLM(args=args, prompts=clustered_prompt, schema=WidthClusterListSchema, max_new_tokens=3000, json_mode=True, temperature=0.6, top_p=0.99)[0]
            if type(cluster_topics) == str:
                cluster_outputs = json.loads(clean_json_string(cluster_topics)) if "```" in cluster_topics else json.loads(cluster_topics.strip())
            else:
                cluster_outputs = [json.loads(clean_json_string(c)) if "```" in c else json.loads(c.strip()) for c in cluster_topics]
            
            # cluster_outputs = json.loads(clean_json_string(cluster_topics)) if "```" in cluster_topics else json.loads(cluster_topics.strip())
            success = True
        except Exception as e:
            success = False
            print(f'failed width expansion clustering attempt #{attempts}!')
            print(cluster_topics)
            print(str(e))
        attempts += 1
    
    args.llm = 'vllm'
    
    if not success:
        print(f'FAILED WIDTH EXPANSION!')
        return []
    
    print('clusters:\n', cluster_outputs)
    cluster_outputs = cluster_outputs['new_cluster_topics']
    final_expansion = []
    dim = node.dimension

    for subtopic_cluster in cluster_outputs:
        sibling_label = subtopic_cluster[f"label"]
        sibling_desc = subtopic_cluster[f"description"]
        mod_key = sibling_label.replace(' ', '_').lower()
        mod_full_key = sibling_label.replace(' ', '_').lower() + f"_{dim}"
        
        if mod_full_key not in label2node:
            child_node = Node(
                    id=len(id2node),
                    label=mod_key,
                    dimension=dim,
                    description=sibling_desc,
                    parents=[node],
                    source='width'
                )
            node.add_child(mod_key, child_node)
            id2node[child_node.id] = child_node
            label2node[mod_full_key] = child_node
            final_expansion.append(mod_key)
        elif label2node[mod_full_key] in label2node[node.label + f"_{dim}"].get_ancestors():
            continue
        else:
            child_node = label2node[mod_full_key]
            node.add_child(mod_key, child_node)
            child_node.add_parent(node)
            final_expansion.append(mod_key)
    
    if len(final_expansion) == 0:
        print(f"NOTICE!!!! {cluster_outputs}")

    return final_expansion




######################## DEPTH EXPANSION ########################

def expandNodeDepth(args, node, id2node, label2node):
    node_ancestors = node.get_ancestors()
    if node_ancestors is None:
        ancestors = "None"
    else:
        node_ancestors.reverse()
        ancestors = " -> ".join([ancestor.label for ancestor in node_ancestors])
    
    # identify potential subtopic options from list of papers
    args.llm = 'vllm'
    subtopic_prompts = [constructPrompt(args, depth_system_instruction, depth_main_prompt(paper, node, ancestors)) 
                   for paper in node.papers.values()]
    subtopic_outputs = promptLLM(args=args, prompts=subtopic_prompts, schema=DepthExpansionSchema, max_new_tokens=300, json_mode=True, temperature=0.6, top_p=0.99)

    subtopic_outputs = [json.loads(clean_json_string(c))['new_subtopic_label'].replace(' ', '_').lower() 
                   if "```" in c else json.loads(c.strip())['new_subtopic_label'].replace(' ', '_').lower() 
                   for c in subtopic_outputs]
    
    subtopic_outputs = [w for w in subtopic_outputs if w + f"_{node.dimension}" not in label2node]

    if len(subtopic_outputs) == 0:
        return []
    
    freq_options = dict(Counter(subtopic_outputs))

    print(freq_options)

    all_node_labels = ", ".join(list(label2node.keys()))

    args.llm = 'gpt'

    prompts = [constructPrompt(args, depth_cluster_system_instruction, depth_cluster_main_prompt(freq_options, node, ancestors, all_node_labels))]

    success = False
    attempts = 0
    while (not success) and (attempts < 5):
        try:
            cluster_topics = promptLLM(args=args, prompts=prompts, schema=DepthClusterListSchema, max_new_tokens=3000, json_mode=True, temperature=0.6, top_p=0.99)[0]
            if type(cluster_topics) == str:
                cluster_outputs = json.loads(clean_json_string(cluster_topics)) if "```" in cluster_topics else json.loads(cluster_topics.strip())
            else:
                cluster_outputs = [json.loads(clean_json_string(c)) if "```" in c else json.loads(c.strip()) for c in cluster_topics]
            # cluster_outputs = json.loads(clean_json_string(cluster_topics)) if "```" in cluster_topics else json.loads(cluster_topics.strip())
            success = True
        except Exception as e:
            success = False
            print(f'failed depth expansion attempt #{attempts}!')
            print(cluster_outputs)
            print(str(e))

        attempts += 1
    if not success:
        print(f'FAILED DEPTH EXPANSION!')
        return [], False

    final_expansion = []
    cluster_outputs = cluster_outputs['new_cluster_topics']
    dim = node.dimension

    for subtopic_cluster in cluster_outputs:
        child_label = subtopic_cluster[f"label"].replace(' ', '_').lower()
        child_desc = subtopic_cluster[f"description"]
        child_full_label = child_label + f"_{dim}"

        if child_label == node.dimension:
          continue
        if child_full_label not in label2node:
            child_node = Node(
                    id=len(id2node),
                    label=child_label,
                    dimension=dim,
                    description=child_desc,
                    parents=[node],
                    source='depth'
                )
            node.add_child(child_label, child_node)
            id2node[child_node.id] = child_node
            label2node[child_full_label] = child_node
            final_expansion.append(child_label)
        elif label2node[child_full_label] in label2node[node.label + f"_{dim}"].get_ancestors():
            continue
        else:
            child_node = label2node[child_full_label]
            node.add_child(child_label, child_node)
            child_node.add_parent(node)
            final_expansion.append(child_label)

    return final_expansion, True


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()