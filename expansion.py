from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from collections import deque
from taxonomy import Node
import json
from utils import clean_json_string
from model_definitions import constructPrompt, promptLLM
from prompts import CandidateSchema
import argparse
import os

def llmExpansion(args, taxo, model, temperature=0.5, top_p=0.99):
    queue = deque([taxo.root])

    expansion_prompts = []
    focus_nodes =  []

    while queue:
        curr_node = queue.popleft()
        focus_nodes.append(curr_node)

        # if it is not a leaf node, perform width expansion on its children
        if len(curr_node.children):
            queue.extend(curr_node.children)
            path = " -> ".join(curr_node.path[1:])
            sibs = ", ".join(map(str, curr_node.children))
            parent_sibs = ", ".join(map(str, [n.label for n in curr_node.parents[0].children if n != curr_node]))

            init_prompt = 'You are an assistant that performs width expansion of taxonomies. Width expansion in taxonomies adds more nodes at the same level, increasing nodes adjacent to a parent node\'s existing children (sibling nodes) without adding depth. For example, expanding a taxonomy of NLP tasks from [\"text_classification\" and \"named_entity_recognition\"] to include \"machine_translation\", and \"question_answering\" would be a width expansion. On the other hand, \"fine_grained_text_classification\" SHOULD NOT be added as it is at a deeper level compared to \"text_classification\" and instead should be considered its child (depth expansion).'
            # init_prompt = f'''You are an assistant that identifies keywords that could replace all instances of <mask> in a given excerpt.
            # <example>
            # ---
            # The given excerpt can be: "text_classification" is a node within a taxonomy, where the path to it is: nlp_tasks -> text_classification. The children of text_classification are: ["sentiment_analysis", "spam_detection", and <mask>]. <mask> does not fall under any other topic adjacent to text_classification, including machine_translation and question_answering.
            
            # For this excerpt, we can replace <mask> with several options, making our example output:
            
            # {{
            #     "parent_node": "text_classification",
            #     "explanation: "stance_detection, long_document_classification, and 'fine_grained_classification' are all subtopics of parent text_classification and are at the same level of specificity as sentiment_analysis and spam_detection. They are only relevant to text_classification and not any adjacent topics like machine_translation and question_answering.",
            #     "candidate_nodes": ["stance_detection", "long_document_classification", "fine_grained_classification"]
            # }}
            # ---
            # </example>
            # '''
            # main_prompt = f'''
            # Output a diverse list of unique keywords that could replace all instances of <mask> in the following excerpt:
            # ---
            # "{curr_node.label} is a node within a taxonomy, where the path to it is: {curr_node.path[1:]}. The children of {curr_node.label} are: [{sibs}, and <mask>]. <mask> does not fall under any other topic adjacent to {curr_node.label}, including {parent_sibs}."
            # ---
            # Output JSON Format:
            # {{
            #     "parent_node": "{curr_node.label}",
            #     "explanation": <a string explanation of your chosen children of {curr_node.label} and siblings of {sibs}>,
            #     "candidate_nodes": <list of diverse strings that can replace <mask> in the above excerpt>
            # }}'''
            main_prompt = f'''
            Taxonomy Path to Parent Node: {curr_node.path[1:]}
            Parent Node: {curr_node.label}
            Children of Parent Node (Existing Siblings): [{sibs}].
            
            Can you expand the set of existing siblings with several candidate node options? Only the new siblings should be exclusively listed within the key, "candidate_nodes".

            Each new sibling should meet the following constraints:
            1. Be a valid (child) subtopic/subcategory of "{curr_node.label}"
            2. Not be a possible subtopic/subcategory of any of the following topics: {parent_sibs}.
            3. Be a valid (sibling) adjacent topic/category to the existing sibling set: [{sibs}]. Valid means at the same depth as the sibling topics but not similar to the existing siblings.
            4. Be able to replace <mask> in the following statement: "Subtopics of topic {curr_node.label} are: {sibs}, and <mask>."

            Output JSON Format:
            {{
                "parent_node": "{curr_node.label}",
                "explanation": <a string explanation of why you chose the candidate_nodes below as the children of {curr_node.label} and why they are also siblings of {sibs}>,
                "candidate_nodes": <list of strings (minimum 1 string, maximum 20 strings) where values are the new children/subtopics of {curr_node.label} and siblings/adjacent topics (1-3 words) to [{sibs}]>

            }}'''
            expansion_prompts.append(constructPrompt(args, init_prompt, main_prompt))
        # if it is a leaf node, perform depth expansion
        else:
            sibs = ", ".join(map(str, [n.label for n in curr_node.parents[0].children if n != curr_node]))
            path = " -> ".join(curr_node.path[1:])
            init_prompt = 'You are an assistant that performs depth expansion of taxonomies. Depth expansion in taxonomies adds nodes deeper to a given node, these being children concepts/topics which EXCLUSIVELY fall under the specified parent node and not the parent\'s siblings. For example, given a taxonomy of NLP tasks, expanding "text_classification" depth-wise (where its siblings are [\"named_entity_recognition\", \"machine_translation\", and \"question_answering\"]) would create the children nodes, [\"sentiment_analysis\", \"spam_detection\", and \"document_classification\"] (any suitable number of children). On the other hand, \"open_domain_question_answering\" SHOULD NOT be added as it belongs to sibling, \"question_answering\".'

            # init_prompt = f'''You are an assistant that identifies keywords that could replace all instances of <mask> in a given excerpt.
            # <example>
            # ---
            # The given excerpt can be: "text_classification" is a node within a taxonomy, where the path to it is: nlp_tasks -> text_classification. A child of text_classification is <mask>. <mask> is irrelevant to the following nodes: ["named_entity_recognition", "machine_translation", "question_answering"].
            
            # For this excerpt, we can replace <mask> with several options, making our example output:
            
            # {{
            #     "parent_node": "text_classification",
            #     "explanation: "stance_detection, long_document_classification, and 'fine_grained_classification' are all subtopics of parent, text_classification, and are all at the same level of specificity. They are only relevant to text_classification and not any adjacent topics like named_entity_recognition, machine_translation, and question_answering.",
            #     "candidate_nodes": ["stance_detection", "long_document_classification", "fine_grained_classification"]
            # }}
            # ---
            # </example>
            # '''

            # main_prompt = f'''Output a list of diverse keywords that could each replace all instances of <mask> in the following excerpt:
            # ---
            # "{curr_node.label} is a node within a taxonomy, where the path to it is: {path}. A child of {curr_node.label} is <mask>. <mask> is irrelevant to the following nodes: [{sibs}]."
            # ---
            # Output JSON Format:
            # {{
            #     "parent_node": "{curr_node.label}",
            #     "explanation": <a string explanation of your chosen children of {curr_node.label}, irrelevant to {sibs}>,
            #     "candidate_nodes": <list of diverse strings that can replace <mask> in the above excerpt>
            # }}'''
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
                "explanation": <a string explanation behind why you chose the candidate_nodes below as the children of {curr_node.label}, and how they are irrelevant to {sibs}>,
                "candidate_nodes": <list of strings (minimum 1 string, maximum 20 strings) where values are the labels of {curr_node.label}'s new subtopics (1-3 words each) and irrelevant to sibling topics: [{sibs}]>

            }}'''
            expansion_prompts.append(constructPrompt(args, init_prompt, main_prompt))
    
    output_dict = promptLLM(args, expansion_prompts, schema=CandidateSchema, max_new_tokens=1000, temperature=temperature, top_p=top_p)

    candidate_nodes = [json.loads(clean_json_string(c)) if "```json" in c else json.loads(c.strip()) for c in output_dict]
        
    for curr_node, candidates in zip(focus_nodes, candidate_nodes):
        for candidate in candidates["candidate_nodes"]:
            candidate = candidate.lower().replace(' ', '_')
            if candidate not in taxo.label2id:
                node_id = str(len(taxo.id2label))
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