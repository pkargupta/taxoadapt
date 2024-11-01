from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
from taxonomy import Node
import json
from utils import clean_json_string
from model_definitions import constructPrompt, promptLlamaVLLM
from prompts import CandidateSchema
import argparse
import os

def llmExpansion(taxo, model, temperature=0.5):
    queue = deque([taxo.root])

    expansion_prompts = []
    focus_nodes =  []

    while queue:
        curr_node = queue.popleft()
        focus_nodes.append(curr_node)

        # if it is not a leaf node, perform width expansion on its children
        if len(curr_node.children):
            queue.extend(curr_node.children)
            sibs = ", ".join(map(str, curr_node.children))
            init_prompt = 'You are an assistant that performs width expansion of taxonomies. Width expansion in taxonomies adds more nodes at the same level, increasing nodes adjacent to a parent node\'s existing children (sibling nodes) without adding depth. For example, expanding a taxonomy of NLP tasks from [\"text_classification\" and \"named_entity_recognition\"] to include \"machine_translation\", and \"question_answering\" would be a width expansion. On the other hand, \"fine_grained_text_classification\" SHOULD NOT be added as it is at a deeper level compared to \"text_classification\" and instead should be considered its child (depth expansion).'
            main_prompt = f'''
            Taxonomy Path to Parent Node: {curr_node.path[1:]}
            Parent Node: {curr_node.label}
            Children of Parent Node (Existing Siblings): [{sibs}].
            
            Can you expand the set of existing siblings with several candidate node options? Only the new siblings should be exclusively listed within the key, "candidate_nodes".

            Each new sibling should meet the following constraints:
            1. Be a valid (child) subtopic/subcategory of "{curr_node.label}"
            2. Be a valid (sibling) adjacent topic/category to the existing sibling set: [{sibs}]. Valid means at the same depth as the sibling topics but not similar to the existing siblings.
            3. Be able to replace <mask> in the following statement: "Subtopics of topic {curr_node.label} are: {sibs}, and <mask>."

            Output JSON Format:
            {{
                "parent_node": "{curr_node.label}",
                "candidate_nodes": <list of strings where values are the new children/subtopics of {curr_node.label} and siblings/adjacent topics (1-3 words) to [{sibs}]>

            }}'''
            expansion_prompts.append(constructPrompt(init_prompt, main_prompt, api=False))
        # if it is a leaf node, perform depth expansion
        else:
            sibs = ", ".join(map(str, [n.label for n in curr_node.parents[0].children if n != curr_node]))

            init_prompt = 'You are an assistant that performs depth expansion of taxonomies. Depth expansion in taxonomies adds nodes deeper to a given node, these being children concepts/topics which EXCLUSIVELY fall under the specified parent node and not the parent\'s siblings. For example, given a taxonomy of NLP tasks, expanding "text_classification" depth-wise (where its siblings are [\"named_entity_recognition\", \"machine_translation\", and \"question_answering\"]) would create the children nodes, [\"sentiment_analysis\", \"spam_detection\", and \"document_classification\"] (any suitable number of children). On the other hand, \"open_domain_question_answering\" SHOULD NOT be added as it belongs to sibling, \"question_answering\".'
            main_prompt = f'''
            Taxonomy Path to Parent Node: {curr_node.path[1:]}
            Parent Node: {curr_node.label}
            Siblings of Parent Node: [{sibs}].
            
            Can you generate a diverse set of candidate children nodes for the parent node? These should be exclusively listed within the key, "candidate_nodes".

            Each new child should meet the following constraints:
            1. Be a valid child (subtopic/subcategory) of "{curr_node.label}". Valid means at a deeper/finer level of specificity than the parent node and cannot fall under the existing siblings.
            2. Unique and diverse from the other new children.

            Output JSON Format:
            {{
                "parent_node": "{curr_node.label}",
                "candidate_nodes": <list of strings where values are the labels of {curr_node.label}'s new subtopics (1-3 words each) and irrelevant to sibling topics: [{sibs}]>

            }}'''
            expansion_prompts.append(constructPrompt(init_prompt, main_prompt, api=False))
    
    output_dict = promptLlamaVLLM(expansion_prompts, schema=CandidateSchema, max_new_tokens=500, temperature=temperature)

    candidate_nodes = [json.loads(clean_json_string(c)) if "```json" in c else json.loads(c.strip()) for c in output_dict]
        
    for curr_node, candidates in zip(focus_nodes, candidate_nodes):
        for candidate in candidates["candidate_nodes"]:
            if candidate not in taxo.label2id:
                node_id = len(taxo.id2label)
                candidate_node = Node(node_id, candidate, description=None, level=curr_node.level+1)
                candidate_node.addParent(curr_node)
                curr_node.addChild(candidate_node)
                taxo.id2label[node_id] = candidate
                taxo.label2id[candidate] = node_id
                taxo.vocab['phrases'][candidate] = model.encode(candidate.replace('_', ' '))
    

    return expansion_prompts, candidate_nodes



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', type=str, help='dataset directory')
    parser.add_argument('--dataset', type=str, help='dataset name')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--lr', default=5e-5, type=float)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()