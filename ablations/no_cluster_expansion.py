import json
import argparse
from pydantic import BaseModel, StringConstraints
from typing_extensions import Annotated
from typing import Dict
from collections import Counter

from ..taxonomy import Node
from ..utils import clean_json_string
from ..model_definitions import constructPrompt, promptLLM

dimension_definitions = {
    'tasks': """Task: we assume that all papers are associated with a specific task(s). Always output "Task" as one of the paper types unless you are absolutely sure the paper does not address any task.""",
    'methodologies': """Methodology: a paper that introduces, explains, or refines a method or approach, providing theoretical foundations, implementation details, and empirical evaluations to advance the state-of-the-art or solve specific problems.""",
    'datasets': """Datasets: introduces a new dataset, detailing its creation, structure, and intended use, while providing analysis or benchmarks to demonstrate its relevance and utility. It focuses on advancing research by addressing gaps in existing datasets/performance of SOTA models or enabling new applications in the field.""",
    'evaluation_methods': """Evaluation Methods: a paper that assesses the performance, limitations, or biases of models, methods, or datasets using systematic experiments or analyses. It focuses on benchmarking, comparative studies, or proposing new evaluation metrics or frameworks to provide insights and improve understanding in the field.""",
    'real_world_domains': """Real-World Domains: demonstrates the use of techniques to solve specific, real-world problems or address specific domain challenges. It focuses on practical implementation, impact, and insights gained from applying methods in various contexts. Examples include: product recommendation systems, medical record summarization, etc."""
    }

node_dimension_definitions = {
    'tasks': """Defines and categorizes research efforts aimed at solving specific problems or objectives within a given field, such as classification, prediction, or optimization.""",
    'methodologies': """Types of techniques, models, or approaches used to address various challenges, including algorithmic innovations, frameworks, and optimization strategies.""",
    'datasets': """Types of methods to structure data collections used in research, including ways to curate or analyze datasets, detailing their properties, intended use, and role in advancing the field.""",
    'evaluation_methods': """Types of methods for assessing the performance of models, datasets, or techniques, including new metrics, benchmarking techniques, or comparative performance studies.""",
    'real_world_domains': """Types of practical or industry-specific domains in which techniques and methodologies can be applied, exploring implementation, impact, and challenges of real-world problems."""
}


class WidthExpansionSchema(BaseModel):
	new_subtopic_label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]
	new_subtopic_description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]
	

width_system_instruction = """You are an assistant that is performing taxonomy width expansion, which is defined as the process of increasing the number of distinct categories or branches within a taxonomy to capture a broader range of concepts, topics, or entities. It adds new sibling nodes—categories that share the same parent node as existing ones—to broaden the scope of a taxonomy while maintaining its hierarchical structure.

You are provided a list of siblings under a parent node and a paper's title & abstract. What subtopic of the parent node does the paper discuss, which is at the same level of specificity as the existing siblings? By specificity, we mean that your new_subtopic_label and the existing_siblings are "equally specific": the topics are at the same level of detail or abstraction; they are on the same conceptual plane without overlap. In other words, they would be sibling nodes within a topical taxonomy.
"""


def width_main_prompt(paper, node, ancestors, nl='\n'):
	out = f"""
<input>
<parent_node>
{node.label}
</parent_node>
<parent_node_description>
{node.label} is a type of {node.dimension}: {node.description}
</parent_node_description>
<type_definition>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</type_definition>
<path_to_parent_node>
{ancestors}
</path_to_parent_node>

<paper_title>
{paper.title}
</paper_title>

<paper_abstract>
{paper.abstract}
</paper_abstract>

<existing_siblings>
{nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}" for c_label, c in node.get_children().items()])}
</existing_siblings>

</input>

Given the input paper title and abstract, identify its {node.dimension} class label that falls under the parent_node, {node.label}, and is a sibling topic to the existing_siblings. In other words, answer the question: what type of {node.label} {node.dimension} does the paper propose?

Your output should be in the following JSON format:
{{
  "new_subtopic_label": <value type is string; string is a new topic label (a type of {node.dimension}) that is the paper's true primary topic at the same level of depth/specificity as the other class labels in existing_siblings>,
  "new_subtopic_description": <string sub-{node.dimension} sentence-long description>
}}
"""
	return out

width_cluster_system_instruction = """You are an clusterer that is performing taxonomy width expansion, which is defined as the process of increasing the number of distinct categories or branches within a taxonomy to capture a broader range of concepts, topics, or entities. It adds new sibling nodes—categories that share the same parent node as existing ones—to broaden the scope of a taxonomy while maintaining its hierarchical structure.

You are choosing your new sibling topic clusters based on which subtopics are covered by papers that discuss the parent node. Your job is to identify unique clusters formed from the input set of paper topics. For each cluster you identify, you must provide a cluster name (in similar format to the paper_topics) as its key, a 1 sentence description of the cluster name, and a list of all the input paper_topics covered within the cluster. Your new topic clusters should have a topic name that is a sibling topic to the existing_siblings BUT DISTINCT. MAKE SURE EVERY SIBLING HAS THE SAME LEVEL OF GRANULARITY/SPECIFICITY. Also make sure that each of your new sibling topic clusters are UNIQUE; they SHOULD NOT currently exist within the existing set of nodes (existing_nodes)."""

######################## DEPTH EXPANSION ########################

depth_system_instruction = """You are an assistant that is performing taxonomy depth expansion, which is defined as adding subcategory nodes deeper to a given root_topic node, these being children concepts/topics which EXCLUSIVELY fall under the specified parent node and not the parent\'s siblings. For example, given a taxonomy of tasks, expanding "text_classification" depth-wise (where its siblings are [\"named_entity_recognition\", \"machine_translation\", and \"question_answering\"]) would create the children nodes, [\"sentiment_analysis\", \"spam_detection\", and \"document_classification\"] (any suitable number of children). On the other hand, \"open_domain_question_answering\" SHOULD NOT be added as it belongs to sibling, \"question_answering\".

You are provided a parent node and a paper's title & abstract. What subtopic of the parent_node does the paper discuss, which is more specific than the parent node? In other words, they would have a parent-child node relationship within a topical taxonomy.
"""

class DepthExpansionSchema(BaseModel):
	new_subtopic_label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]
	new_subtopic_description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]

def depth_main_prompt(paper, node, ancestors, nl='\n'):
	out = f"""
<input>
<parent_node>
{node.label}
</parent_node>
<parent_node_description>
{node.label} is a type of {node.dimension}: {node.description}
</parent_node_description>
<type_definition>
{node.dimension}: {node_dimension_definitions[node.dimension]}
</type_definition>
<path_to_parent_node>
{ancestors}
</path_to_parent_node>

<paper_title>
{paper.title}
</paper_title>

<paper_abstract>
{paper.abstract}
</paper_abstract>

</input>

Given the input paper title and abstract, identify its {node.dimension} class label that falls under the parent_node, {node.label}. In other words, answer the question: what type of {node.label} {node.dimension} does the paper propose?

Your output should be in the following JSON format:
{{
  "new_subtopic_label": <value type is string; string is a new topic label (a type of {node.dimension}) that is the paper's true primary topic at the deeper/more specific level than {node.label}>,
  "new_subtopic_description": <string sub-{node.dimension} sentence-long description>
}}
"""
	return out


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
	
	subset_unlabeled_papers = list(unlabeled_papers.values())[:5]
	
	args.llm = 'gpt'
	
	exp_prompts = [constructPrompt(args, width_system_instruction, width_main_prompt(paper, node, ancestors)) for paper in subset_unlabeled_papers]
	exp_outputs = promptLLM(args=args, prompts=exp_prompts, schema=WidthExpansionSchema, max_new_tokens=2000, json_mode=True, temperature=0.1, top_p=0.99)
	
	
	exp_outputs = [json.loads(clean_json_string(c)) 
				   if "```" in c else json.loads(c.strip()) 
				   for c in exp_outputs]

	# FILTERING OF EXPANSION OUTPUTS
	
	args.llm = 'vllm'
	
	print('clusters:\n', exp_outputs)
	final_expansion = []
	dim = node.dimension

	for subtopic_opt in exp_outputs:
		sibling_label = subtopic_opt[f"new_subtopic_label"]
		sibling_desc = subtopic_opt[f"new_subtopic_description"]
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
		print(f"NOTICE!!!! {exp_outputs}")

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
	args.llm = 'gpt'
	
	subset_papers = list(node.papers.values())[:5]
	
	subtopic_prompts = [constructPrompt(args, depth_system_instruction, depth_main_prompt(paper, node, ancestors)) 
				   for paper in subset_papers]
	subtopic_outputs = promptLLM(args=args, prompts=subtopic_prompts, schema=DepthExpansionSchema, max_new_tokens=2000, json_mode=True, temperature=0.1, top_p=0.99)

	exp_outputs = [json.loads(clean_json_string(c)) 
				   if "```" in c else json.loads(c.strip()) 
				   for c in subtopic_outputs]

	# FILTERING OF EXPANSION OUTPUTS
	
	args.llm = 'vllm'
	
	print('clusters:\n', exp_outputs)
	final_expansion = []
	dim = node.dimension

	for subtopic_opt in exp_outputs:
		child_label = subtopic_opt[f"new_subtopic_label"]
		child_desc = subtopic_opt[f"new_subtopic_description"]
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