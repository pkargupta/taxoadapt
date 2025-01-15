from taxonomy import Node
import json
from utils import clean_json_string
from model_definitions import constructPrompt, promptLLM
import argparse
from pydantic import BaseModel, StringConstraints
from typing_extensions import Annotated
from typing import Dict


width_system_instruction = """You are an assistant that is provided a list of class labels. You determine whether or not a given paper's primary topic exists within the input class label list. If so, you output that topic label name. If not, then suggest a new topic at the same level of specificity. By specificity, we mean that your new_class_label and the existing_class_options are "equally specific": the topics are at the same level of detail or abstraction; they are on the same conceptual plane without overlap. In other words, they could be sibling nodes within a topical taxonomy.
"""

class WidthExpansionSchema(BaseModel):
  new_class_label: Annotated[str, StringConstraints(strip_whitespace=True)]


def width_main_prompt(paper, node, nl='\n'):
   out = f"""Given the following paper title and abstract, suggest for a new class label to be added to the list of existing_class_options. All of these class labels (existing_class_options and your new class) should fall under the topic: {node.label}.

"Title": "{paper.title}"
"Abstract": "{paper.abstract}"

existing_class_options (list of existing class labels): {"; ".join([f"{c}" for c in node.get_children()])}

Here is some additional information about each existing class option:
{nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}" for c_label, c in node.get_children().items()])}


Your output should be in the following JSON format:
{{
  "new_class_label": <value type is string; string is the new topic label that is the paper's true primary topic at the same level of depth/specificity as the other class labels in existing_class_options>,
}}
"""
   return out

# {nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}Example phrases used in {c_label} papers: {c.phrases}{nl}Example sentences used in {c_label} papers: {c.sentences}" for c_label, c in node.get_children().items()])}

cluster_system_instruction = "You are a clusterer, identifying clusters relevant to being formed from an input set of labels. For each cluster you identify, you must provide a cluster name (in similar format to the input labels) as its key and a 1 sentence description of the cluster name."

class ClusterSchema(BaseModel):
    cluster_topic_description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]

class ClusterListSchema(BaseModel):
    new_cluster_topics: Dict[str, ClusterSchema]



def cluster_main_prompt(options, node, nl='\n'):
   siblings = node.get_children()
   out = f"""Given the following set of candidate node labels, can you identify the most popular, unique but MINIMAL set of clusters that best represent all candidate_node_labels and are highly likely to be siblings of the following existing nodes within a taxonomy (existing_sibling_nodes)? Each new cluster topic that you suggest should fall under the parent topic node, {node.label}, and be a type of {node.dimension}.\n

existing_sibling_nodes: {str(siblings)}

candidate_node_labels:\n{str(options)}


Your output should be in the following JSON format:
{{
  "new_cluster_topics":
  {{
    "<key type is string; string is the first new cluster topic label that is the paper's true primary topic at the same level of depth/specificity as the other class labels in existing_class_options>": {{
      "cluster_topic_description": "<generate a string sentence-long description of your new first cluster topic>"
    }},
    ...,
    "<key type is string; string is the kth (max 10th) new cluster topic label that is the paper's true primary topic at the same level of depth/specificity as the other class labels in existing_class_options>": {{
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

    if len(unlabeled_papers) < args.min_density:
        return [] 
    
    exp_prompts = [constructPrompt(args, width_system_instruction, width_main_prompt(paper, node)) for paper in unlabeled_papers.values()]
    exp_outputs = promptLLM(args=args, prompts=exp_prompts, schema=WidthExpansionSchema, max_new_tokens=300, json_mode=True, temperature=0.1, top_p=0.99)
    exp_outputs = [json.loads(clean_json_string(c))['new_class_label'].replace(' ', '_').lower() if "```" in c else json.loads(c.strip())['new_class_label'].replace(' ', '_').lower() for c in exp_outputs]

    # FILTERING OF EXPANSION OUTPUTS

    width_options = list(set(exp_outputs))
    args.llm = 'gpt'
    clustered_prompt = [constructPrompt(args, cluster_system_instruction, cluster_main_prompt(width_options, node))]
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
                    parents=[node]
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

    return final_expansion


depth_system_instruction = """You are an assistant that performs depth expansion of taxonomies. Depth expansion in taxonomies adds subcategory nodes deeper to a given root_topic node, these being children concepts/topics which EXCLUSIVELY fall under the specified parent node and not the parent\'s siblings. For example, given a taxonomy of NLP tasks, expanding "text_classification" depth-wise (where its siblings are [\"named_entity_recognition\", \"machine_translation\", and \"question_answering\"]) would create the children nodes, [\"sentiment_analysis\", \"spam_detection\", and \"document_classification\"] (any suitable number of children). On the other hand, \"open_domain_question_answering\" SHOULD NOT be added as it belongs to sibling, \"question_answering\".
"""

class NodeSchema(BaseModel):
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]

class NodeListSchema(BaseModel):
    root_topic: Dict[str, NodeSchema]


def depth_main_prompt(node):
   ancestors = ", ".join([ancestor.label for ancestor in node.get_ancestors()])

   out = f"""Your root_topic is: {node.label}\nA subcategory is a specific division within a broader category that organizes related items or concepts more precisely. Output up to 5 children, subcategories of {node.dimension} that fall under {node.label} and generate corresponding sentence-long descriptions for each. Make sure each type is unique to the topics: {node.label}, {ancestors}. 

Output your taxonomy ONLY in the following JSON format, replacing each label name key with its correct subcategory label name:\n
{{
  root_topic:
  {{
    "<label name of your first sub-category>": {{
      "description": "<generate a string description of your subcategory>"
    }},
    ...,
    "<label name of your kth (max 5th) sub-category>": {{
      "description": "<generate a string description of subtask_k>"
    }},
  }}
}}
"""
   return out

def expandNodeDepth(args, node, id2node, label2node):
    
    prompts = [constructPrompt(args, depth_system_instruction, depth_main_prompt(node))]

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
                    parents=[node]
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