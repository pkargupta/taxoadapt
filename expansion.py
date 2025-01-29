from taxonomy import Node
import json
from utils import clean_json_string
from model_definitions import constructPrompt, promptLLM
import argparse
from pydantic import BaseModel, StringConstraints
from typing_extensions import Annotated
from typing import Dict
from collections import Counter


code_width_instruction = lambda node, candidate_subtopics: f"""You are attempting to identify subtopics for parent topic, {node.label}, that best represent and partition a pool of papers.
The parent topic already has the following {node.dimension} subtopics (existing_subtopics):

existing_subtopics: {"; ".join([f"{c}" for c in node.get_children()])}

You have the following candidate subtopics with their corresponding number of papers:

{candidate_subtopics}


Given the above set of candidate subtopics as reference, can you identify the non-overlapping cluster subtopics of parent {node.dimension} topic {node.label} that best represent and partition all of the candidates above (maximize the number of papers that are mapped to each). They should all be siblings of each other and the existing_subtopics (same level of depth/specificity) within the taxonomy (no cluster subtopic should fall under another cluster subtopic)? Each new cluster topic that you suggest should be a more specific subtopic under the parent topic node, {node.label}, and be a type of task. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily."""

code_depth_instruction = lambda node, candidate_subtopics: f"""You are attempting to identify subtopics for parent topic, {node.label}, that best represent and partition a pool of papers.
You have the following candidate subtopics with their corresponding number of papers:

{candidate_subtopics}


Given the above set of candidate subtopics as reference, can you identify the non-overlapping cluster subtopics of parent {node.dimension} topic {node.label} that best represent and partition all of the candidates above (maximize the number of papers that are mapped to each). They should all be siblings of each other (same level of depth/specificity) within the taxonomy (no cluster subtopic should fall under another cluster subtopic)? Each new cluster topic that you suggest should be a more specific subtopic under the parent topic node, {node.label}, and be a type of task. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily."""

code_prompt = lambda node: f"""
Treat this as a quantitative reasoning (optimization) problem. Select subtopics that MINIMIZE the TOTAL NUMBER of subtopics needed yet simultaneously MAXIMIZE the number of total papers mapped. In the "subtopic_reasoning", explain your quantitative reasoning, using the candidate subtopics as variables with their integer values equal to the number of papers mapped to the respective topics.

Use code to show your quantitative work.

Output your final answer in following XML and JSON format:

<code>
<include all of your code for your quantitative reasoning here>
</code>

<subtopic_json>
{{
    "subtopics_of_{node.label}": [
        {{
        "mapped_papers": <integer value; using the candidate subtopics as variables with the number of papers mapped to them as their integer values, compute the number of papers mapped to this subtopic>
         "subtopic_label": <string value; 2-5 word subtopic label (a type of task)>,
         "subtopic_description": <string value; sentence-long description of subtopic>
        }},
        ...
    ]
}}
</subtopic_json>
"""



width_system_instruction = """You are an assistant that is provided a list of class labels. You determine whether or not a given paper's primary topic exists within the input class label list. If it does exist, output that topic label name. If not, then suggest a new topic at the same level of specificity. By specificity, we mean that your class_label and the existing_class_options are "equally specific": the topics are at the same level of detail or abstraction; they are on the same conceptual plane without overlap. In other words, they could be sibling nodes within a topical taxonomy.
"""

class WidthExpansionSchema(BaseModel):
  class_label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]


def width_main_prompt(paper, node, nl='\n'):
   out = f"""Given the following paper title and abstract, identify its {node.dimension} class label from the existing_class_options or, if it does not exist, suggest a new {node.dimension} class_label that falls under {node.label} ({node.description}). IF you suggest a new class_label, it should be at a similar topical level to the existing_class_options (e.g., subtopics of {node.label} are: {", ".join([f"{c}" for c in node.get_children()])}, and <class_label>).

"Title": "{paper.title}"
"Abstract": "{paper.abstract}"

existing_class_options (list of existing class labels): {"; ".join([f"{c}" for c in node.get_children()])}

Here is some additional information about each existing class option:
{nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}" for c_label, c in node.get_children().items()])}


Your output should be in the following JSON format:
{{
  "class_label": <value type is string; string is either an existing_class from existing_class_options or the new topic label (a type of {node.dimension}) that is the paper's true primary topic at the same level of depth/specificity as the other class labels in existing_class_options>,
}}
"""
   return out

# {nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}Example phrases used in {c_label} papers: {c.phrases}{nl}Example sentences used in {c_label} papers: {c.sentences}" for c_label, c in node.get_children().items()])}

cluster_system_instruction = "You are a clusterer, identifying unique clusters formed from an input set of labels. For each cluster you identify, you must provide a cluster name (in similar format to the input labels) as its key and a 1 sentence description of the cluster name."

class ClusterSchema(BaseModel):
    cluster_topic_description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]

class ClusterListSchema(BaseModel):
    new_cluster_topics: Dict[str, ClusterSchema]



def cluster_main_prompt(options, node, nl='\n'):
   siblings = node.get_children()
   print(options)
   out = f"""Below is a dictionary of candidate node labels, where each key is the candidate node label and value is number of papers which are mapped to that candidate node:

candidate_node_labels:\n{str(options)}


Given the above set of candidate node labels, can you identify the non-overlapping clusters that best represent and partition all of candidate_node_labels (maximize the number of papers that are mapped to each). They should all be siblings (same level of depth/specificity) of the following existing nodes within a taxonomy (existing_sibling_nodes)? Each new cluster topic that you suggest should be a more specific subtopic under the parent topic node, {node.label}, and be a type of {node.dimension}. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily.\n

existing_sibling_nodes: {str(siblings)}


Your output should be in the following JSON format:
{{
  "new_cluster_topics":
  {{
    "<key type is string; string is the first new cluster topic label (a type of {node.dimension}) based on the candidate_node_labels and is at the same level of depth/specificity as the other class labels in existing_class_options>": {{
      "cluster_topic_description": "<generate a string sentence-long description of your new first cluster topic>"
    }},
    ...,
    "<key type is string; string is the kth (max 5th) new cluster topic label (a type of {node.dimension}) based on the candidate_node_labels and is at the same level of depth/specificity as the other class labels in existing_class_options>": {{
      "cluster_topic_description": "<generate a string sentence-long description of your new kth cluster topic>"
    }},
  }}
}}
"""
   return out


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

    print(f'node {node.label} ({node.dimension}) has {len(unlabeled_papers)} unlabeled papers!')

    if len(unlabeled_papers) <= args.max_density:
        return [] 
    
    exp_prompts = [constructPrompt(args, width_system_instruction, width_main_prompt(paper, node)) for paper in unlabeled_papers.values()]
    exp_outputs = promptLLM(args=args, prompts=exp_prompts, schema=WidthExpansionSchema, max_new_tokens=300, json_mode=True, temperature=0.1, top_p=0.99)
    exp_outputs = [json.loads(clean_json_string(c))['class_label'].replace(' ', '_').lower() 
                   if "```" in c else json.loads(c.strip())['class_label'].replace(' ', '_').lower() 
                   for c in exp_outputs]
    
    exp_outputs = [w for w in exp_outputs if w + f"_{node.dimension}" not in label2node]
    if len(exp_outputs) == 0:
        return []
    freq_options = Counter(exp_outputs)

    # FILTERING OF EXPANSION OUTPUTS
    args.llm = 'gpt'
    clustered_prompt = [constructPrompt(args, cluster_system_instruction, cluster_main_prompt(freq_options, node))]
    success = False
    attempts = 0
    while (not success) and (attempts < 5):
        try:
            cluster_topics = promptLLM(args=args, prompts=clustered_prompt, schema=ClusterListSchema, max_new_tokens=3000, json_mode=True, temperature=0.1, top_p=0.99)[0]
            cluster_outputs = json.loads(clean_json_string(cluster_topics)) if "```" in cluster_topics else json.loads(cluster_topics.strip())
            success = True
        except Exception as e:
            success = False
            print(f'failed clustering attempt #{attempts}!')
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

    for key, value in cluster_outputs.items():
        sibling_label = key
        sibling_desc = value['cluster_topic_description'] if 'cluster_topic_description' in value else value['description']
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


depth_system_instruction = """You are an assistant that performs depth expansion of taxonomies. Depth expansion in taxonomies adds subcategory nodes deeper to a given root_topic node, these being children concepts/topics which EXCLUSIVELY fall under the specified parent node and not the parent\'s siblings. For example, given a taxonomy of NLP tasks, expanding "text_classification" depth-wise (where its siblings are [\"named_entity_recognition\", \"machine_translation\", and \"question_answering\"]) would create the children nodes, [\"sentiment_analysis\", \"spam_detection\", and \"document_classification\"] (any suitable number of children). On the other hand, \"open_domain_question_answering\" SHOULD NOT be added as it belongs to sibling, \"question_answering\".
"""

class NodeSchema(BaseModel):
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]

class NodeListSchema(BaseModel):
    root_topic: Dict[str, NodeSchema]


def depth_main_prompt(node, ancestors, subtopics):

   out = f"""Your parent node (root_topic) is a type of {node.dimension}: {node.label}\nWe define {node.label} as: {node.description}. The path to parent node, "{node.label}", in the taxonomy is: {ancestors}.\nA subtopic is a specific division within a broader category that organizes related items or concepts more precisely. 

Below is a dictionary of candidate subtopics of {node.label}, where each key is the candidate subtopic and the value is the number of papers which fall under that subtopic:

{subtopics}


Given the above set of candidate subtopics as reference, can you identify the non-overlapping cluster subtopics of parent topic "{node.label}" that best represent and partition all of candidates above (maximize the number of papers that are mapped to each). They should all be siblings (same level of depth/specificity) within the taxonomy (no cluster subtopic should fall under another cluster subtopic)? Each new cluster topic that you suggest should be a more specific subtopic under the parent topic node, {node.label}, and be a type of {node.dimension}. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily.\n

Generate corresponding sentence-long descriptions for each cluster topic. 

Output your taxonomy ONLY in the following JSON format, replacing each label name key with its correct subcategory label name:\n
{{
  root_topic:
  {{
    "<label name of your first cluster subtopic (a type of {node.dimension})>": {{
      "description": "<generate a string description of your cluster subtopic>"
    }},
    ...,
    "<label name of your kth (max 5th) cluster subtopic (a type of {node.dimension})>": {{
      "description": "<generate a string description of cluster subtopic_k>"
    }},
  }}
}}
"""
   return out

subtopic_system_instruction = lambda node, ancestors: f"""You are an assistant that is provided with a {node.label} paper's title and abstract. We define {node.label} as {node.description}, where the path of ancestors to reach {node.label} is: {ancestors}. Your task is to identify the subtopic discussed by the paper that falls under the parent topic, {node.label}. DO NOT JUST OUTPUT THE PARENT TOPIC {node.label}."""

class SubtopicSchema(BaseModel):
  subtopic: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]


def subtopic_main_prompt(paper, node, nl='\n'):
   out = f"""What {node.dimension} subtopic of parent topic, {node.label}, is the following paper title and abstract discussing?

"Title": "{paper.title}"
"Abstract": "{paper.abstract}"

Your output should be in the following JSON format:
{{
  "subtopic": <value type is string; the string's value is the paper's subtopic label UNDER {node.label} (a type of {node.dimension}; lowercase, underscore format: subtopic_name)>,
}}
"""
   return out

def expandNodeDepth(args, node, id2node, label2node):
    node_ancestors = node.get_ancestors()
    if node_ancestors is None:
        ancestors = "None"
    else:
        node_ancestors.reverse()
        ancestors = " -> ".join([ancestor.label for ancestor in node_ancestors])
    
    # identify potential subtopic options from list of papers
    args.llm = 'vllm'
    subtopic_prompts = [constructPrompt(args, subtopic_system_instruction(node, ancestors), subtopic_main_prompt(paper, node)) 
                   for paper in node.papers.values()]
    subtopic_outputs = promptLLM(args=args, prompts=subtopic_prompts, schema=SubtopicSchema, max_new_tokens=300, json_mode=True, temperature=0.1, top_p=0.99)

    subtopic_outputs = [json.loads(clean_json_string(c))['subtopic'].replace(' ', '_').lower() 
                   if "```" in c else json.loads(c.strip())['subtopic'].replace(' ', '_').lower() 
                   for c in subtopic_outputs]
    
    subtopic_outputs = [w for w in subtopic_outputs if w + f"_{node.dimension}" not in label2node]

    if len(subtopic_outputs) == 0:
        return []
    
    freq_options = Counter(subtopic_outputs)

    args.llm = 'gpt'

    prompts = [constructPrompt(args, depth_system_instruction, depth_main_prompt(node, ancestors, freq_options))]

    success = False
    attempts = 0
    while (not success) and (attempts < 5):
        try:
            outputs = promptLLM(args=args, prompts=prompts, schema=NodeListSchema, max_new_tokens=3000, json_mode=True, temperature=0.1, top_p=0.99)[0]
            gen_child = json.loads(clean_json_string(outputs)) if "```" in outputs else json.loads(outputs.strip())
            gen_child = gen_child['root_topic'] if 'root_topic' in gen_child else gen_child[node.label]
            success = True
        except Exception as e:
            success = False
            print(outputs)
            print(f'failed depth expansion attempt #{attempts}!')
            print(str(e))

        attempts += 1
    if not success:
        print(f'FAILED DEPTH EXPANSION!')
        return [], False

    final_expansion = []
    dim = node.dimension

    for key, value in gen_child.items():
        child_label = key.replace(' ', '_').lower()
        child_full_label = child_label + f"_{dim}"
        child_desc = value['description']
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