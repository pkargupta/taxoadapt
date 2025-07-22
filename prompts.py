from pydantic import BaseModel, conset, conlist, StringConstraints, Field
from typing_extensions import Annotated
from typing import Dict

class NodeSchema(BaseModel):
    description: Annotated[str, StringConstraints(strip_whitespace=True)]


class NodeListSchema(BaseModel):
    # argument_list : conlist(argument_schema, min_length=1,max_length=10)
    root_topic: Dict[str, NodeSchema]

class EnrichSchema(BaseModel):
    node_to_enrich: Annotated[str, StringConstraints(strip_whitespace=True)]
    id: Annotated[str, StringConstraints(strip_whitespace=True)]
    commonsense_key_phrases: conset(str, min_length=20, max_length=50)
    commonsense_sentences: conset(str, min_length=10, max_length=50)
# NLP
dimension_definitions = {
    'tasks': """Task: we assume that all papers are associated with a specific task(s). Always output "Task" as one of the paper types unless you are absolutely sure the paper does not address any task.""",
    'methodologies': """Methodology: a paper that introduces, explains, or refines a method or approach, providing theoretical foundations, implementation details, and empirical evaluations to advance the state-of-the-art or solve specific problems.""",
    'datasets': """Datasets: introduces a new dataset, detailing its creation, structure, and intended use, while providing analysis or benchmarks to demonstrate its relevance and utility. It focuses on advancing research by addressing gaps in existing datasets/performance of SOTA models or enabling new applications in the field.""",
    'evaluation_methods': """Evaluation Methods: a paper that assesses the performance, limitations, or biases of models, methods, or datasets using systematic experiments or analyses. It focuses on benchmarking, comparative studies, or proposing new evaluation metrics or frameworks to provide insights and improve understanding in the field.""",
    'real_world_domains': """Real-World Domains: demonstrates the use of techniques to solve specific, real-world problems or address specific domain challenges. It focuses on practical implementation, impact, and insights gained from applying methods in various contexts. Examples include: product recommendation systems, medical record summarization, etc."""
    }

# bio
# dimension_definitions = {
#     'experimental_methods': """Experimental Methods: a paper that introduces, explains, or significantly refines experimental techniques, protocols, laboratory methods, or biological assays, providing detailed descriptions and validations to improve accuracy, reproducibility, or insight in biological research.""",

#     'datasets': """Datasets: a paper that introduces new biological datasets (e.g., genomic sequences, imaging data, ecological observations), detailing their generation, structure, annotation, and intended use, and provides initial analyses or benchmarks demonstrating their value in addressing gaps or enabling new biological insights.""",

#     'theoretical_advances': """Theoretical Advances: a paper that proposes new biological theories, models, frameworks, or conceptual insights, supported by rigorous analysis, modeling, or experimental validation, aimed at improving fundamental understanding of biological systems.""",
    
#     'applications': """Applications: a paper which demonstrates practical use of biological knowledge or techniques to address real-world problems in domains such as biomedicine (therapies, diagnostics, drug development), agriculture (crop improvement, pest control), or conservation (species protection, ecosystem management), focusing on practical impact and applied outcomes.""",

#     'evaluation_methods': """Evaluation Methods: a paper which systematically evaluates biological techniques, datasets, or computational methods, using benchmarking, comparative analyses, or novel evaluation metrics, to provide deeper insights into their effectiveness, limitations, or biases, thereby enhancing understanding and guiding future research."""
# }

# NLP
node_dimension_definitions = {
    'tasks': """Defines and categorizes research efforts aimed at solving specific problems or objectives within a given field, such as classification, prediction, or optimization.""",
    'methodologies': """Types of techniques, models, or approaches used to address various challenges, including algorithmic innovations, frameworks, and optimization strategies.""",
    'datasets': """Types of methods to structure data collections used in research, including ways to curate or analyze datasets, detailing their properties, intended use, and role in advancing the field.""",
    'evaluation_methods': """Types of methods for assessing the performance of models, datasets, or techniques, including new metrics, benchmarking techniques, or comparative performance studies.""",
    'real_world_domains': """Types of practical or industry-specific domains in which techniques and methodologies can be applied, exploring implementation, impact, and challenges of real-world problems."""
}
# node_dimension_definitions = {
#     'experimental_methods': """Types of experimental techniques, protocols, laboratory procedures, or biological assays introduced or significantly refined, detailing their design, validation, and implementation to improve accuracy, reproducibility, or effectiveness in biological research.""",

#     'datasets': """Types of biological datasets introduced (e.g., genomic, proteomic, imaging, ecological data), describing their creation, structure, annotation, and intended use, accompanied by initial analyses or benchmarks demonstrating their utility in enabling novel insights or addressing research gaps.""",

#     'theoretical_advances': """Types of new biological theories, conceptual frameworks, or models proposed, supported by rigorous analytical, mathematical, or empirical validation, aimed at enhancing fundamental understanding of biological systems or phenomena.""",
    
#     'applications': """Types of practical applications of biological research or techniques in domains such as biomedicine (therapeutics, diagnostics), agriculture (crop improvement, pest management), or conservation biology (species protection, ecosystem management), emphasizing real-world impact, feasibility, and applied outcomes.""",

#     'evaluation_methods': """Types of systematic approaches for evaluating biological methods, datasets, or computational techniques through benchmarking, comparative analysis, or novel performance metrics, aimed at identifying strengths, weaknesses, biases, or effectiveness, thus informing and guiding future research directions."""
# }

  
def multi_dim_prompt(node):
    topic = node.label
    ancestors = ", ".join([ancestor.label for ancestor in node.get_ancestors()])

    system_instruction = f'You are a helpful assistant that constructs taxonomies for a given root topic: {topic} {"" if ancestors == "" else " (relevant to" + ancestors + ")"} (types of {node.dimension}). Keep in mind that research papers will be mapped to the nodes within your constructed taxonomy. We define {node.dimension} below:\n{dimension_definitions[node.dimension]}\n'

    main_prompt = f'Your root_topic is: {topic}\nA subcategory is a specific division within a broader category that organizes related items or concepts more precisely. Output up to 5 children, subcategories that are types of {node.dimension} which fall under {topic} and generate corresponding sentence-long descriptions for each. Make sure each type is unique to the topics: {topic}, {ancestors}.'

    if ('domain' in node.dimension) or ('application' in node.dimension):
        main_prompt += f'\n Remember that {node.dimension} means a real-world domain category in which a paper can be applied to (for example, news or science could be a subcategory of {node.dimension}).'

    json_output_format = f'''Output your taxonomy ONLY in the following JSON format, replacing each label name key with its correct subcategory label name:\n
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
}}'''
    
    return system_instruction, main_prompt, json_output_format


type_cls_system_instruction = """You are a helpful multi-label classification assistant which helps me label papers based on their paper type. They may be more than one.

Paper types (type:definition):

1. Task: we assume that all papers are associated with a specific task(s). Always output "Task" as one of the paper types unless you are absolutely sure the paper does not address any task.
2. Methodology: a paper that introduces, explains, or refines a method or approach, providing theoretical foundations, implementation details, and empirical evaluations to advance the state-of-the-art or solve specific problems. 
3. Datasets: introduces a new dataset, detailing its creation, structure, and intended use, while providing analysis or benchmarks to demonstrate its relevance and utility. It focuses on advancing research by addressing gaps in existing datasets/performance of SOTA models or enabling new applications in the field. 
4. Evaluation Methods: a paper that assesses the performance, limitations, or biases of models, methods, or datasets using systematic experiments or analyses. It focuses on benchmarking, comparative studies, or proposing new evaluation metrics or frameworks to provide insights and improve understanding in the field. 
5. Real-World Domains: demonstrates the use of techniques to solve specific, real-world problems or address specific domain challenges. It focuses on practical implementation, impact, and insights gained from applying methods in various contexts. Examples include: product recommendation systems, medical record summarization, etc.
"""

# type_cls_system_instruction = """You are a helpful multi-label classification assistant which helps me label papers based on their paper type. They may be more than one.

# Paper types (type:definition):

# 1. Experimental Methods: a paper which introduces, explains, or significantly refines experimental techniques, protocols, laboratory methods, or biological assays, providing detailed descriptions and validations to improve accuracy, reproducibility, or insight in biological research.
# 2. Datasets: a paper that introduces new biological datasets (e.g., genomic sequences, imaging data, ecological observations), detailing their generation, structure, annotation, and intended use, and provides initial analyses or benchmarks demonstrating their value in addressing gaps or enabling new biological insights.
# 3. Theoretical Advances: a paper which proposes new biological theories, models, frameworks, or conceptual insights, supported by rigorous analysis, modeling, or experimental validation, aimed at improving fundamental understanding of biological systems. 
# 4. Applications: a paper that demonstrates practical use of biological knowledge or techniques to address real-world problems in domains such as biomedicine (therapies, diagnostics, drug development), agriculture (crop improvement, pest control), or conservation (species protection, ecosystem management), focusing on practical impact and applied outcomes. 
# 5. Evaluation Methods: A paper which systematically evaluates biological techniques, datasets, or computational methods, using benchmarking, comparative analyses, or novel evaluation metrics, to provide deeper insights into their effectiveness, limitations, or biases, thereby enhancing understanding and guiding future research.
# """

class TypeClsSchema(BaseModel):
  tasks: bool
  methodologies: bool
  datasets: bool
  evaluation_methods: bool
  real_world_domains: bool

# class TypeClsSchema(BaseModel):
#   experimental_methods: bool
#   datasets: bool
#   theoretical_advances: bool
#   applications: bool
#   evaluation_methods: bool

def type_cls_main_prompt(paper):
   out = f"""Given the following paper title and abstract, can you output a Pythonic list of all paper type labels relevant to this paper. 

"Title": "{paper.title}"
"Abstract": "{paper.abstract}"

Your output should be in the following JSON format:
{{
  "tasks": True,
  "methodologies": <return True if the paper is a Methodology paper, False otherwise>,
  "datasets": <return True if the paper is a Dataset paper, False otherwise>,
  "evaluation_methods": <return True if the paper is an Evaluation paper, False otherwise>,
  "real_world_domains": <return True if the paper is a Real-World Domain/Application paper, False otherwise>,
}}
"""
   return out

# def type_cls_main_prompt(paper):
#    out = f"""Given the following paper title and abstract, can you output a Pythonic list of all paper type labels relevant to this paper. 

# "Title": "{paper.title}"
# "Abstract": "{paper.abstract}"

# Your output should be in the following JSON format:
# {{
#   "experimental_methods": <return True if the paper focuses on an experimental method, False otherwise>,
#   "datasets": <return True if the paper is a Dataset paper, False otherwise>,
#   "theoretical_advances": <return True if the paper focuses on advancing theory, False otherwise>,
#   "applications": <return True if the paper is a paper focused on application, False otherwise>,
#   "evaluation_methods": <return True if the paper focuses on an evaluation method, False otherwise>,
# }}
# """
#    return out


def baseline_prompt(paper, node):
   
   cats = "\n".join([f"{node.description}" for c in node.children])

   return f'''You will be provided with a research paper title and abstract. Please select the categories that this paper should be placed under. We provide the list of categories and their respective descriptions. Just give the category names as shown in the list.

title: {paper.title}
abstract: {paper.abstract}

categories:
{cats}
'''

init_enrich_prompt = "You are a helpful assistant that performs taxonomy enrichment using realistic specific keywords and sentences that would be used in research papers. These realistic keywords and sentences will be used to identify research papers which discuss a specific taxonomy node."

init_classify_prompt = "You are a helpful assistant that identifies the class labels for the provided research paper, performing multi-label classification."

bulk_enrich_prompt = lambda dict_str: f'''I am providing you a JSON which contains a taxonomy detailing concepts for research papers. Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases" and "example_sentences" fields for each concept node (enrichment of both the root node and its children/descendants) that contains these fields? The required fields are already present for you, so you do not need to create any new keys for concepts without them. Here are the instructions for each field under concept A:

1. "example_key_phrases": This is a list (Python-formatted) of 20 key phrases (e.g., SUBTOPICS of the given concept node A) commonly used amongst research papers that EXCLUSIVELY DISCUSS that concept node (concept A's key phrases/subtopics should be highly relevant to its concept A's parent concept, and NOT OR RARELY be mentioned in ANY of its SIBLING concepts B; A and B share the same parent concept). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., "key_phrase"). Each key phrase should be unique.
2. "example_sentences": This is a list (Python-formatted) of 10 key sentences that could be used to discuss the concept node A within an research paper. These key sentences should be SPECIFIC, not generic, to Concept A (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts. Utilize your knowledge of the concept node A (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.

---
input_taxo:
{dict_str}
---

Given the input taxonomy JSON above, output your enriched taxonomy JSON (only output your final JSON, no other information) following the above rules and taxonomy rules in general. Your entire response/output is going to consist of a single JSON object {{}}, and you will NOT wrap it within JSON Markdown markers. ONLY output your final enriched taxonomy JSON, no additional explanations or comments.

Your output format should be:
---
output_taxo:
[output taxonomy JSON]
---
'''

one_shot = f'''
Example Input:
Here is an example output for the topic, 'multi_agent_reinforcement_learning', a subtopic of topic 'reinforcement learning'. The terms are irrelevant to the sibling subtopics, 'hierarchical_reinforcement_learning' and 'meta_reinforcement_learning'.

Example Output:
{{
    "node_to_enrich: "multi_agent_reinforcement_learning",
    "description": "Strategies for reinforcement learning where multiple agents interact within the same environment, learning to collaborate or compete.",
    "example_key_phrases": ["cooperative_learning", "competitive_learning", "nash_equilibrium", "joint_policy_learning", "decentralized_control", "communication_protocols", "agent_modeling", "self_play", "multi_agent_coordination", "teamwork", "reward_allocation", "multi_agent_credit_assignment", "partially_observable_joint_policy", "equilibrium_dynamics", "multi_agent_exploration", "adversarial_agents", "learning_with_sparse_rewards", "role_assignment", "emergent_behavior", "heterogeneous_agent_interaction"],
    "example_sentences": ["in multi_agent_reinforcement_learning, agents must adapt to dynamic environments by leveraging decentralized_control strategies, ensuring robustness in the absence of centralized oversight.", "cooperative_learning allows agents to share information efficiently, optimizing joint_policy_learning to achieve a shared goal within multi_agent_coordination frameworks.", "competitive_learning agents often employ adversarial_agents techniques, balancing self-play with learning_with_sparse_rewards to improve their strategies over time.", "achieving nash_equilibrium in multi_agent interactions can help stabilize the learning process, especially when heterogeneous_agent_interaction is present.", "communication_protocols are essential in ensuring seamless information flow, particularly when reward_allocation is based on shared global goals.", "multi_agent_credit_assignment remains a challenge, as agents must determine their contribution to the overall task without centralized feedback.", "emergent_behavior can arise in complex systems where role_assignment evolves dynamically during cooperative and competitive tasks.", "agent_modeling plays a crucial role in predicting the actions of other agents, especially when equilibrium_dynamics shift as policies evolve.", "agents operating in partially_observable_joint_policy environments must rely on incomplete information, making learning robust communication_protocols essential.", "in multi_agent_exploration scenarios, agents must balance the need to gather new information with the exploitation of current knowledge to improve coordination."]
}}
'''

parent_prompt = lambda taxo, node: f" and is the subtopic of all of the following topics: [{', '.join(taxo.get_par(node.node_id, node=False))}]" if node.parents[0].node_id != -1 else ""

main_simple_enrich_prompt = lambda taxo, node, sibs: f'''"{node.label}" is a topic in {taxo.root.label}{parent_prompt(taxo, node)}. Please generate 20 realistic key terms and sentences about the '{node.label}' topic that are relevant to '{node.label}' but irrelevant to the topics: {sibs}. The terms should be short (1-3 words), concise, and distinctive to {node.label}. The sentences should have specific details and resemble realistic sentences found in {taxo.root.label} research papers.

Your output format should be in the following JSON format (where node_to_enrich, id and description for {node.label} should match their respective values in the input taxonomy 'input_taxo' aka COPIED OVER FROM input_taxo):
---
{{
    "node_to_enrich: "{node.label}"
    "id": "{node.node_id}",
    "description": "{node.description if node.description else '<string where value is a 1-sentence description of node_to_enrich>'}",
    "example_key_phrases": <list of 20 diverse, short terms where values are realistic and relevant to '{node.label}' and DISSIMILAR to any of the following: '{sibs}'>,
    "example_sentences": <list of 10 diverse sentences where values are sentences used in papers about '{node.label}' and DISSIMILAR to any of the following: '{sibs}'>
}}
---
'''

main_long_enrich_prompt = lambda node, sibs, dict_str: f'''I am providing you a JSON which contains a taxonomy detailing concepts in research papers (tag 'input_taxo'). Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases" and "example_sentences" fields for the specified node (tag 'node_to_enrich')? A research paper relevant to 'node_to_enrich' will be relevant to all concept nodes present in the taxonomy path to the node, 'node_to_enrich', as listed in 'path_to_node'. Here are your instructions on how to enrich the fields for node, 'node_to_enrich':

1. "example_key_phrases": This is a list (Python-formatted) of 20 key, realistic phrases (e.g., SUBTOPICS of the given 'node_to_enrich') commonly written within research papers that EXCLUSIVELY DISCUSS 'node_to_enrich'. 'node_to_enrich's key phrases/subtopics should be highly relevant to all of 'node_to_enrich's ancestors listed in 'path_to_node', and NOT be relevant to ANY other non-ancestor or siblings of 'node_to_enrich' (siblings of 'node_to_enrich' are specified in tag 'siblings' below). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., "key_phrase"). Each key phrase should be unique.
2. "example_sentences": This is a list (Python-formatted) of 10 key, realistic sentences that could be written in a research paper to discuss 'node_to_enrich'. These key sentences should be SPECIFIC, not generic, to 'node_to_enrich' (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts (tag 'siblings'). Utilize your knowledge of the 'node_to_enrich' (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.

---
node_to_enrich: {node.label}
node_to_enrich_id: {node.node_id}
path_to_node: {'->'.join(node.path)}
siblings: {sibs}
input_taxo:
{dict_str}
---

Given the input taxonomy JSON above and the node_to_enrich, output your enrichment of 'node_to_enrich' in JSON format (only output your final JSON, no other information) following the above rules and taxonomy rules in general. Your entire response/output is going to consist of a single JSON object {{}}, and you will NOT wrap it within JSON Markdown markers. ONLY output your final enriched node_to_enrich JSON, no additional explanations or comments.

Your output format should be in the following JSON format (where node_to_enrich, id and description for {node.label} should match their respective values in the input taxonomy 'input_taxo' aka COPIED OVER FROM input_taxo):
---
{{
    "node_to_enrich: "{node.label}"
    "id": "{node.node_id}",
    "description": "{node.description if node.description else '<string where value is a 1-sentence description of node_to_enrich based on its path, path_to_node>'}",
    "example_key_phrases": <list of 20 diverse strings where values are realistic and relevant key phrases/subtopics to 'node_to_enrich' and DISSIMILAR to any 'siblings'>,
    "example_sentences": <list of 10 diverse strings where values are sentences used in papers about 'node_to_enrich' and DISSIMILAR to any 'siblings'>
}}
---
'''

main_classify_prompt = lambda node, paper: f'''Given the 'title', 'abstract', and 'content' (provided below) of a research paper that uses large language models for graphs, select the class labels (tag 'class_options') that should be assigned to this paper (multi-label classification). If the research paper SHOULD NOT be labeled with any of the classes in 'class_options', then output an empty list. We provide additional descriptions (tag 'class_descriptions') for each class option for your reference.

---
paper_id: {paper.id}
title: {paper.title}
abstract: {paper.abstract}
content: {paper.content[:10000]}
class_options (class id: class label name): {"; ".join([f"{c.node_id}: {c.label}" for c in node.children])}
class_descriptions: {"; ".join([f"{c.label}: {c.description}" for c in node.children])}
---

Your output format should be in the following JSON format:
---
{{
    paper_id: {paper.id}
    class_options: {[c.node_id for c in node.children]}
    class_labels: <list of ints where values are the class ids (options provided above in 'class_options') that the paper should be labeled with>
}}
---
'''

main_enrich_prompt_paper = lambda node, dict_str: f'''I am providing you a JSON which contains a taxonomy detailing concepts for "using llms on graphs" in research papers (tag 'input_taxo'). Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases", "example_sentences", "example_paper_titles", and "example_paper_abstracts" fields for the specified node (tag 'node_to_enrich')? A research paper relevant to 'node_to_enrich' will be relevant to all concept nodes present in the path to 'node_to_enrich', as listed in 'path_to_node'. Here are your instructions on how to enrich the fields for node, 'node_to_enrich':

1. "example_key_phrases": This is a list (Python-formatted) of 20 key, realistic phrases (e.g., SUBTOPICS of the given concept node A) commonly written within research papers that EXCLUSIVELY DISCUSS that concept node (node A's key phrases/subtopics should be highly relevant to its node A's parent concept, and NOT be relevant to ANY other non-ancestor or descendants of node A; A and B share the same parent concept). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., "key_phrase"). Each key phrase should be unique.
2. "example_sentences": This is a list (Python-formatted) of 10 key, realistic sentences that could be written in an research paper to discuss the concept node A. These key sentences should be SPECIFIC, not generic, to node A (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts. Utilize your knowledge of the concept node A (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.
3. "example_paper_titles": This is a list (Python-formatted) of 2 realistic, detailed titles of a research paper that discusses 'node_to_enrich', as well as all nodes in 'path_to_node'. Each title should not be able to be placed under any other node within 'input_taxo' that is not an ancestor or descendant of 'node_to_enrich'. Each title should correspond to the abstract with the same index in "example_paper_abstracts".
4. "example_paper_abstracts": This is a list (Python-formatted) of 2 realistic, detailed research paper abstracts (paragraph long) that discusses 'node_to_enrich', as well as all nodes in 'path_to_node'. Each abstract should not be able to be placed under any other node within 'input_taxo' that is not an ancestor or descendant of 'node_to_enrich'. Each abstract should correspond to the title with the same index in "example_paper_titles".

---
node_to_enrich: {node.label}
path_to_node: {'->'.join(node.path)}
input_taxo:
{dict_str}
---

Given the input taxonomy JSON above and the node_to_enrich, output your enrichment of 'node_to_enrich' in JSON format (only output your final JSON, no other information) following the above rules and taxonomy rules in general. Your entire response/output is going to consist of a single JSON object {{}}, and you will NOT wrap it within JSON Markdown markers. ONLY output your final enriched node_to_enrich JSON, no additional explanations or comments.

Your output format should be in the following JSON format (where node_to_enrich, id and description for {node.label} should match their respective values in the input taxonomy 'input_taxo' aka COPIED OVER FROM input_taxo):
---
{{
    "node_to_enrich: "{node.label}"
    "id": "{node.node_id}",
    "description": "{node.description}",
    "example_key_phrases": <list of strings where values are realistic and relevant key phrases/subtopics>,
    "example_sentences": <list of strings where values are sentences>
    "example_paper_titles": <list of strings where values are realstic research paper titles that are relevant to 'node_to_enrich' and 'path_to_node'>
    "example_paper_abstracts": <list of strings where values are realstic research paper abstracts that are relevant to 'node_to_enrich' and 'path_to_node'>
}}
---
'''

class CommonSenseSchema(BaseModel):
    node_to_enrich: Annotated[str, StringConstraints(strip_whitespace=True)]
    id: Annotated[str, StringConstraints(strip_whitespace=True)]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]
    example_key_phrases: conset(str, min_length=20, max_length=50)
    example_sentences: conset(str, min_length=10, max_length=50)
    # example_paper_titles: conset(str, min_length=5)
    # example_paper_abstracts: conset(str, min_length=5)

class SiblingSchema(BaseModel):
    new_siblings: conset(str, min_length=1, max_length=10)
  
class CandidateSchema(BaseModel):
    parent_node: Annotated[str, StringConstraints(strip_whitespace=True)]
    explanation: Annotated[str, StringConstraints(strip_whitespace=True)]
    candidate_nodes: conset(str, min_length=1, max_length=20)

class ClassifySchema(BaseModel):
    paper_id: Annotated[int, Field(strict=True, gt=-1)]
    class_options: conset(int, min_length=1, max_length=100)
    class_labels: conset(int, min_length=0, max_length=10)

str_schema = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.com/product.schema.json",
  "title": "Taxonomy Research Concept",
  "description": "A research concept present within the input taxonomy, input_taxo",
  "type": "object",
  "properties": {
    "id": {
      "description": "The unique identifier for the taxonomy concept node. Already provided within input taxonomy.",
      "type": "string"
    },
    "description": {
      "description": "The description of the taxonomy concept node. Already provided within input taxonomy.",
      "type": "string"
    },
    "example_key_phrases": {
      "description": "20 key phrases (e.g., SUBTOPICS of the given concept node A) commonly used amongst research papers that EXCLUSIVELY DISCUSS that concept node (node A's key phrases/subtopics should be highly relevant to its node A's parent concept, and NOT be relevant to ANY siblings of node A; node A and sibling B share the same parent concept). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., 'key_phrase'). Each key phrase should be unique.",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": True
    },
    "example_sentences": {
      "description": "10 key sentences that could be used to discuss the concept node A within a research paper. These key sentences should be SPECIFIC, not generic, to node A (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts. Utilize your knowledge of the concept node A (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": True
    }
  }
}

quant_width_instruction = lambda node, candidate_subtopics, existing: f"""You are attempting to identify subtopics for parent topic, {node.label}, that best represent and partition a pool of papers. A subtopic is a specific division within a broader category that organizes related items or concepts more precisely.
We define {node.label} (a type of {node.dimension}) as: {node.description}

The parent topic already has the following {node.dimension} subtopics (existing_subtopics), so your chosen subtopics should expand upon the current list:

existing_subtopics: {existing}

You have the following candidate subtopics with their corresponding number of papers:

{candidate_subtopics}


Given the above set of candidate subtopics as reference, can you identify the non-overlapping cluster subtopics of parent {node.dimension} topic {node.label} that best represent and partition all of the candidates above (maximize the number of papers that are mapped to each). They should all be siblings of each other and the existing_subtopics (same level of depth/specificity) within the taxonomy (no cluster subtopic should fall under another cluster subtopic)? Each new cluster topic that you suggest should be a more specific subtopic under the parent topic node, {node.label}, and be a type of task. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily.

Treat this as a quantitative clustering (optimization) problem. Select subtopics that MINIMIZE the TOTAL NUMBER of subtopics needed yet simultaneously MAXIMIZE the number of total papers mapped (where the maximum value possible is the total number of papers which do not fall under existing_subtopics). In the tags <quantitative_reasoning></quantitative_reasoning>, explain your quantitative reasoning, using the candidate subtopics as variables with their integer values equal to the number of papers mapped to the respective topics. REMEMBER that all candidate subtopics under a cluster you form SHOULD BE RELATED.

Here are two example inputs and outputs:

<example_1>
<example_input>
Parent Topic: text_classification
Parent Topic Dimension Type: Task
existing_subtopics: named_entity_recognition


Input Candidate Dictionary:
{{
  "sentiment_analysis": 10,
  "spam_detection": 3,
  "emotion_detection": 2,
  "news_type_classification": 4,
  "junk_mail_detection": 10,
  "social_media_sentiment_detection": 2,
  "event_classification": 8,
  "entity_classification": 20
}}
</example_input>

<example_quantitative_reasoning>
Step 1: Define Variables
x_1: sentiment_analysis (10)
x_2: spam_detection (3)
x_3: emotion_detection (2)
x_4: news_type_classification (4)
x_5: junk_mail_detection (10)
x_6: social_media_sentiment_detection (2)
x_7: event_classification (8)
x_8: entity_classification (20) # we are not considering entity_classification in any clusters since it is already contained within an existing subtopic, named_entity_recognition.

Step 2: Semantic Clustering
Cluster 1 (S_1): x_1 (sentiment_analysis), x_3 (emotion_detection), x_6 (social_media_sentiment_detection). Theme: Sentiment-related classification tasks.
Cluster 2 (S_2): x_2 (spam_detection), x_5 (junk_mail_detection). Theme: Detection of unwanted messages.
Cluster 3 (S_3): x_4 (news_type_classification), x_7 (event_classification). Theme: Classification of events or news categories.

Step 3: Aggregate Frequencies
F_1 = 10 + 2 + 2 = 14 -> Label: "sentiment_analysis"
F_2 = 3 + 10 = 13 -> Label: "spam_detection"
F_3 = 4 + 8 = 12 -> Label: "event_classification"

Step 4: Optimization Result
Total papers covered: 14 + 13 + 12 = 39 (matches the total input papers which do not fall under existing_subtopics).
Reduced variables: 8 -> 3.

</example_quantitative_reasoning>

<example_final_output>
{{
  "subtopics_of_text_classification": [
    {{
      "mapped_papers": 14,
      "subtopic_label": "sentiment_analysis",
      "subtopic_description": "A task focused on identifying sentiment, emotion, or social media sentiment in text."
    }},
    {{
      "mapped_papers": 13,
      "subtopic_label": "spam_detection",
      "subtopic_description": "A task focused on detecting unwanted messages, including spam and junk mail."
    }},
    {{
      "mapped_papers": 12,
      "subtopic_label": "event_classification",
      "subtopic_description": "A task focused on classifying news articles or events in text."
    }}
  ]
}}
</example_final_output>

</example_1>

<example_2>
<example_input>
Parent Topic: statistical_approaches
Parent Topic Dimension Type: Methodologies
existing_subtopics: probabilistic_modeling

Input Candidate Dictionary:
{{
    "hidden_markov_models": 15,
    "naive_bayes_classification": 12,
    "conditional_random_fields": 8,
    "bayesian_networks": 9,
    "maximum_entropy_models": 6,
    "latent_dirichlet_allocation": 10,
    "gaussian_mixture_models": 7,
    "markov_chain_monte_carlo": 5
}}
</example_input>

<example_quantitative_reasoning>
Step 1: Define Variables
x_1: hidden_markov_models (15)
x_2: naive_bayes_classification (12) # we are not considering naive_bayes_classification in any clusters since it is already contained within an existing subtopic, probabilistic_modeling.
x_3: conditional_random_fields (8)
x_4: bayesian_networks (9) # we are not considering bayesian_networks in any clusters since it is already contained within an existing subtopic, probabilistic_modeling.
x_5: maximum_entropy_models (6) # we are not considering maximum_entropy_models in any clusters since it is already contained within an existing subtopic, probabilistic_modeling.
x_6: latent_dirichlet_allocation (10)
x_7: gaussian_mixture_models (7)
x_8: markov_chain_monte_carlo (5)

Step 2: Semantic Clustering
Cluster 1 (S_1): x_1 (hidden_markov_models), x_3 (conditional_random_fields), x_8 (markov_chain_monte_carlo). Theme: Sequence modeling and Markov-based approaches.
Cluster 2 (S_2): x_6 (latent_dirichlet_allocation), x_7 (gaussian_mixture_models). Theme: Topic modeling and mixture models.

Step 3: Aggregate Frequencies
F_1 = 15 + 8 + 5 = 28 -> Label: "sequence_modeling"
F_2 = 10 + 7 = 17 -> Label: "topic_modeling"

Step 4: Optimization Result
Total papers covered: 28 + 17 = 63 (matches the total input papers which do not fall under existing_subtopics).
Reduced variables: 8 -> 2.
</example_quantitative_reasoning>

<example_final_output>
{{
    "subtopics_of_statistical_approaches": [
        {{
            "mapped_papers": 28,
            "subtopic_label": "sequence_modeling",
            "subtopic_description": "A task focused on modeling sequences using Markov-based approaches, such as hidden Markov models and conditional random fields."
        }},
        {{
            "mapped_papers": 17,
            "subtopic_label": "topic_modeling",
            "subtopic_description": "A task focused on identifying topics in text using mixture models, such as latent Dirichlet allocation and Gaussian mixture models."
        }}
    ]
}}
</example_final_output>
</example_2>
"""

quant_depth_instruction = lambda node, candidate_subtopics, ancestors: f"""You are attempting to identify subtopics for parent topic, {node.label}, that best represent and partition a pool of papers. A subtopic is a specific division within a broader category that organizes related items or concepts more precisely.
We define {node.label} (a type of {node.dimension}) as: {node.description}
The path to parent node, "{node.label}", in the taxonomy is: {ancestors}.

You have the following candidate subtopics with their corresponding number of papers:

{candidate_subtopics}


Given the parent node, an input set of candidate subtopics and their dimension type (e.g., either a task, methodology, dataset, real_world_domain, or evaluation_method) as reference, can you identify the non-overlapping cluster subtopics of parent {node.dimension} topic {node.label} that best represent and partition all of the candidates above (maximize the number of papers that are mapped to each). They should all be siblings of each other (same level of depth/specificity) within the taxonomy (no cluster subtopic should fall under another cluster subtopic)? Each new cluster topic that you suggest should be a more specific subtopic under the parent topic node, {node.label}, and be a type of task. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily.

Treat this as a quantitative clustering (optimization) problem. Select subtopics that MINIMIZE the TOTAL NUMBER of subtopics needed yet simultaneously MAXIMIZE the number of total papers mapped (where the maximum value possible is the total number of papers). In the tags <quantitative_reasoning></quantitative_reasoning>, explain your quantitative reasoning, using the candidate subtopics as variables with their integer values equal to the number of papers mapped to the respective topics. REMEMBER that all candidate subtopics under a cluster you form SHOULD BE RELATED.

Here are three input and output examples:

<example_1>
<example_input>
Parent Topic: text_classification
Parent Topic Dimension Type: Task
Input Candidate Dictionary:
{{
  "sentiment_analysis": 10,
  "spam_detection": 3,
  "emotion_detection": 2,
  "news_type_classification": 4,
  "junk_mail_detection": 10,
  "social_media_sentiment_detection": 2,
  "event_classification": 8,
  "named_entity_recognition": 20
}}
</example_input>

<example_quantitative_reasoning>
Step 1: Define Variables
x_1: sentiment_analysis (10)
x_2: spam_detection (3)
x_3: emotion_detection (2)
x_4: news_type_classification (4)
x_5: junk_mail_detection (10)
x_6: social_media_sentiment_detection (2)
x_7: event_classification (8)
x_8: named_entity_recognition (20)

Step 2: Semantic Clustering
Cluster 1 (S_1): x_1 (sentiment_analysis), x_3 (emotion_detection), x_6 (social_media_sentiment_detection). Theme: Sentiment-related classification tasks.
Cluster 2 (S_2): x_2 (spam_detection), x_5 (junk_mail_detection). Theme: Detection of unwanted messages.
Cluster 3 (S_3): x_4 (news_type_classification), x_7 (event_classification). Theme: Classification of events or news categories.
Cluster 4 (S_4): x_8 (named_entity_recognition). Theme: Entity recognition (no overlap with other labels).

Step 3: Aggregate Frequencies
F_1 = 10 + 2 + 2 = 14 -> Label: "sentiment_analysis"
F_2 = 3 + 10 = 13 -> Label: "spam_detection"
F_3 = 4 + 8 = 12 -> Label: "event_classification"
F_4 = 20 -> Label: "named_entity_recognition"

Step 4: Optimization Result
Total papers covered: 14 + 13 + 12 + 20 = 59 (matches total input papers).
Reduced variables: 8 -> 4.

</example_quantitative_reasoning>

<example_final_output>
{{
  "subtopics_of_text_classification": [
    {{
      "mapped_papers": 20,
      "subtopic_label": "named_entity_recognition",
      "subtopic_description": "A task focused on identifying named entities such as people, organizations, or locations in text."
    }},
    {{
      "mapped_papers": 14,
      "subtopic_label": "sentiment_analysis",
      "subtopic_description": "A task focused on identifying sentiment, emotion, or social media sentiment in text."
    }},
    {{
      "mapped_papers": 13,
      "subtopic_label": "spam_detection",
      "subtopic_description": "A task focused on detecting unwanted messages, including spam and junk mail."
    }},
    {{
      "mapped_papers": 12,
      "subtopic_label": "event_classification",
      "subtopic_description": "A task focused on classifying news articles or events in text."
    }}
  ]
}}
</example_final_output>

</example_1>

<example_2>
<example_input>
Parent Topic: text_classification
Parent Topic Dimension Type: dataset

Input Candidate Dictionary:
{{
    "IMDB_reviews": 15,
    "Yelp_reviews": 10,
    "AG_News": 8,
    "20_Newsgroups": 5,
    "MIMIC_III_notes": 12,
    "PubMed_abstracts": 7,
    "Twitter_sentiment": 6,
    "Reddit_comments": 4,
    "SpamAssassin": 10
}}
</example_input>

<example_quantitative_reasoning>
Step 1: Define Variables
x_1: IMDB_reviews (15)
x_2: Yelp_reviews (10)
x_3: AG_News (8)
x_4: 20_Newsgroups (5)
x_5: MIMIC_III_notes (12)
x_6: PubMed_abstracts (7)
x_7: Twitter_sentiment (6)
x_8: Reddit_comments (4)
x_9: SpamAssassin (10)

Step 2: Semantic Clustering
Cluster 1 (S_1): x_1 (IMDB_reviews), x_2 (Yelp_reviews), x_7 (Twitter_sentiment), x_8 (Reddit_comments). Theme: Sentiment analysis datasets from reviews or social media.
Cluster 2 (S_2): x_3 (AG_News), x_4 (20_Newsgroups). Theme: News article classification datasets.
Cluster 3 (S_3): x_5 (MIMIC_III_notes), x_6 (PubMed_abstracts). Theme: Medical text classification datasets.
Cluster 4 (S_4): x_9 (SpamAssassin). Theme: Spam detection dataset (no overlap with other domains).

Step 3: Aggregate Frequencies
F_1 = 15 + 10 + 6 + 4 = 35 -> Label: "sentiment_analysis_datasets"
F_2 = 8 + 5 = 13 -> Label: "news_classification_datasets"
F_3 = 12 + 7 = 19 -> Label: "medical_text_datasets"
F_4 = 10 -> Label: "spam_detection_datasets"

Step 4: Optimization Result
Total papers covered: 35 + 13 + 19 + 10 = 77 (matches total input papers).
Reduced variables: 9 -> 4.
</example_quantitative_reasoning>

<example_final_output>
{{
    "subtopics_of_text_classification": [
        {{
            "mapped_papers": 35,
            "subtopic_label": "sentiment_analysis_datasets",
            "subtopic_description": "Datasets containing reviews or social media text labeled for sentiment analysis tasks."
        }},
        {{
            "mapped_papers": 19,
            "subtopic_label": "medical_text_datasets",
            "subtopic_description": "Datasets comprising clinical notes or scientific abstracts for medical text classification."
        }},
        {{
            "mapped_papers": 13,
            "subtopic_label": "news_classification_datasets",
            "subtopic_description": "Datasets with news articles categorized by topic or event for classification tasks."
        }},
        {{
            "mapped_papers": 10,
            "subtopic_label": "spam_detection_datasets",
            "subtopic_description": "Datasets focused on identifying spam or unwanted content in text."
        }}
    ]
}}
</example_final_output>
</example_2>

<example_3>
<example_input>
Parent Topic: deep_learning_approaches
Parent Topic Dimension Type: Methodologies
Input Candidate Dictionary:
{{
    "convolutional_neural_networks": 12,
    "recurrent_neural_networks": 18,
    "long_short_term_memory": 15,
    "gated_recurrent_units": 10,
    "transformers": 25,
    "attention_mechanisms": 20,
    "autoencoders": 8,
    "generative_adversarial_networks": 7,
    "bert": 22,
    "gpt": 18
}}
</example_input>

<example_quantitative_reasoning>
Step 1: Define Variables
x_1: convolutional_neural_networks (12)
x_2: recurrent_neural_networks (18)
x_3: long_short_term_memory (15)
x_4: gated_recurrent_units (10)
x_5: transformers (25)
x_6: attention_mechanisms (20)
x_7: autoencoders (8)
x_8: generative_adversarial_networks (7)
x_9: bert (22)
x_10: gpt (18)

Step 2: Semantic Clustering
Cluster 1 (S_1): x_1 (convolutional_neural_networks). Theme: Convolutional-based deep learning approaches.
Cluster 2 (S_2): x_2 (recurrent_neural_networks), x_3 (long_short_term_memory), x_4 (gated_recurrent_units). Theme: Recurrent-based deep learning approaches.
Cluster 3 (S_3): x_5 (transformers), x_6 (attention_mechanisms), x_9 (bert), x_10 (gpt). Theme: Transformer and attention-based deep learning approaches.
Cluster 4 (S_4): x_7 (autoencoders), x_8 (generative_adversarial_networks). Theme: Generative deep learning approaches.

Step 3: Aggregate Frequencies
F_1 = 12 -> Label: "convolutional_neural_networks"
F_2 = 18 + 15 + 10 = 43 -> Label: "recurrent_based_approaches"
F_3 = 25 + 20 + 22 + 18 = 85 -> Label: "transformer_based_approaches"
F_4 = 8 + 7 = 15 -> Label: "generative_approaches"

Step 4: Optimization Result
Total papers covered: 12 + 43 + 85 + 15 = 155 (matches total input papers).
Reduced variables: 10 -> 4.
</example_quantitative_reasoning>

<example_final_output>
{{
    "subtopics_of_deep_learning_approaches": [
        {{
            "mapped_papers": 85,
            "subtopic_label": "transformer_based_approaches",
            "subtopic_description": "Deep learning approaches leveraging transformers, attention mechanisms, and models like BERT and GPT."
        }},
        {{
            "mapped_papers": 43,
            "subtopic_label": "recurrent_based_approaches",
            "subtopic_description": "Deep learning approaches utilizing recurrent neural networks, LSTMs, and GRUs for sequential data."
        }},
        {{
            "mapped_papers": 15,
            "subtopic_label": "generative_approaches",
            "subtopic_description": "Deep learning approaches focused on generative models, including autoencoders and GANs."
        }},
        {{
            "mapped_papers": 12,
            "subtopic_label": "convolutional_neural_networks",
            "subtopic_description": "Deep learning approaches employing convolutional neural networks for feature extraction."
        }}
    ]
}}
</example_final_output>
</example_3>

"""

quant_width_prompt = lambda node, candidate_subtopics, existing: f"""For your own input, determine the minimal set of subtopics which maximizes the number of papers covered by following the same quantitative reasoning format in <example></example> and include it inside the tags, <quantitative_reasoning></quantitative_reasoning>

<input>
Parent Topic: {node.label}
Parent Topic Dimension Type: {node.dimension}
existing_subtopics: {existing}

Input Candidate Dictionary:
{candidate_subtopics}

</input>

Output your final answer in following XML and JSON format:

<final_output>

<quantitative_reasoning>
<include your quantitative reasoning in the same format as the example here>
</quantitative_reasoning>

<subtopic_json>
{{
    "subtopics_of_{node.label}": [
        {{
        "mapped_papers": <integer value; using the candidate subtopics as variables with the number of papers mapped to them as their integer values, compute the number of papers mapped to this subtopic>
         "subtopic_label": <string value; 2-5 word cluster subtopic label (a type of {node.dimension}) that falls under {node.label} and is at the same level of depth/specificity as {existing}>,
         "subtopic_description": <string value; sentence-long description of cluster subtopic>
        }},
        ...
    ]
}}
</subtopic_json>
</final_output>
"""

quant_depth_prompt = lambda node, candidate_subtopics: f"""For your own input, determine the minimal set of subtopics which maximizes the number of papers covered by following the same quantitative reasoning format in <example></example> and include it inside the tags, <quantitative_reasoning></quantitative_reasoning>

<input>
Parent Topic: {node.label}
Parent Topic Dimension Type: {node.dimension}

Input Candidate Dictionary:
{candidate_subtopics}

</input>

Output your final answer in following XML and JSON format:

<final_output>

<quantitative_reasoning>
<include your quantitative reasoning in the same format as the example here>
</quantitative_reasoning>

<subtopic_json>
{{
    "subtopics_of_{node.label}": [
        {{
        "mapped_papers": <integer value; using the candidate subtopics as variables with the number of papers mapped to them as their integer values, compute the number of papers mapped to this subtopic>
         "subtopic_label": <string value; 2-5 word cluster subtopic label (a type of {node.dimension}) that falls under {node.label}>,
         "subtopic_description": <string value; sentence-long description of cluster subtopic>
        }},
        ...
    ]
}}
</subtopic_json>
</final_output>
"""


######################## WIDTH EXPANSION ########################

width_system_instruction = """You are an assistant that is performing taxonomy width expansion, which is defined as the process of increasing the number of distinct categories or branches within a taxonomy to capture a broader range of concepts, topics, or entities. It adds new sibling nodes—categories that share the same parent node as existing ones—to broaden the scope of a taxonomy while maintaining its hierarchical structure.

You are provided a list of siblings under a parent node and a paper's title & abstract. What subtopic of the parent node does the paper discuss, which is at the same level of specificity as the existing siblings? By specificity, we mean that your new_subtopic_label and the existing_siblings are "equally specific": the topics are at the same level of detail or abstraction; they are on the same conceptual plane without overlap. In other words, they would be sibling nodes within a topical taxonomy.
"""

class WidthExpansionSchema(BaseModel):
  new_subtopic_label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]


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
}}
"""
   return out

width_cluster_system_instruction = """You are an clusterer that is performing taxonomy width expansion, which is defined as the process of increasing the number of distinct categories or branches within a taxonomy to capture a broader range of concepts, topics, or entities. It adds new sibling nodes—categories that share the same parent node as existing ones—to broaden the scope of a taxonomy while maintaining its hierarchical structure.

You are choosing your new sibling topic clusters based on which subtopics are covered by papers that discuss the parent node. Your job is to identify unique clusters formed from the input set of paper topics. For each cluster you identify, you must provide a cluster name (in similar format to the paper_topics) as its key, a 1 sentence description of the cluster name, and a list of all the input paper_topics covered within the cluster. Your new topic clusters should have a topic name that is a sibling topic to the existing_siblings BUT DISTINCT. MAKE SURE EVERY SIBLING HAS THE SAME LEVEL OF GRANULARITY/SPECIFICITY. Also make sure that each of your new sibling topic clusters are UNIQUE; they SHOULD NOT currently exist within the existing set of nodes (existing_nodes)."""

class WidthClusterSchema(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True)]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]
    covered_paper_topics: conlist(str, min_length=1, max_length=20)


class WidthClusterListSchema(BaseModel):
   new_cluster_topics: conlist(WidthClusterSchema, min_length=1, max_length=10)



def width_cluster_main_prompt(options, node, ancestors, all_node_labels, nl='\n'):
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

Your new cluster topics SHOULD NOT BE ANY OF THE FOLLOWING:
<existing_nodes>
{all_node_labels}
</existing_nodes>

<existing_siblings>
{nl.join([f"{c_label}:{nl}{nl}Description of {c_label}: {c.description}{nl}" for c_label, c in node.get_children().items()])}
</existing_siblings>

<paper_topics>
Below is a dictionary of paper topics, where each key is the candidate node label and value is number of papers which are mapped to that candidate node:
candidate_node_labels:\n{str(options)}
</paper_topics>

</input>

What are the primary sub-{node.dimension} topic clusters under the parent_node topic, {node.label}, that would best encompass the above <paper_topics>?
These should be non-overlapping topic clusters that best represent and partition all of paper_topics (maximize the number of papers that are mapped to each). They should all be siblings (same level of depth/specificity) of the existing_siblings within the taxonomy. Each new cluster topic that you suggest should be a more specific subtopic under the parent_node, {node.label}, and be a type of {node.dimension}. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily.\n

Your output should be in the following JSON format with a minimum of one subtopic cluster and a maximum of five:
{{
  "new_cluster_topics":
  [
    {{
    "label": <string sub-{node.dimension} label at the same level of depth/specificity as the other topics in existing_siblings>,
    "description": <string sub-{node.dimension} sentence-long description>,
    "covered_paper_topics": <list of all the input paper_topics covered within this sub-{node.dimension}>
    }},
    ...
  ]
}}

---

Your output JSON:

"""
  return out

######################## DEPTH EXPANSION ########################

depth_system_instruction = """You are an assistant that is performing taxonomy depth expansion, which is defined as adding subcategory nodes deeper to a given root_topic node, these being children concepts/topics which EXCLUSIVELY fall under the specified parent node and not the parent\'s siblings. For example, given a taxonomy of tasks, expanding "text_classification" depth-wise (where its siblings are [\"named_entity_recognition\", \"machine_translation\", and \"question_answering\"]) would create the children nodes, [\"sentiment_analysis\", \"spam_detection\", and \"document_classification\"] (any suitable number of children). On the other hand, \"open_domain_question_answering\" SHOULD NOT be added as it belongs to sibling, \"question_answering\".

You are provided a parent node and a paper's title & abstract. What subtopic of the parent_node does the paper discuss, which is more specific than the parent node? In other words, they would have a parent-child node relationship within a topical taxonomy.
"""

class DepthExpansionSchema(BaseModel):
  new_subtopic_label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=100)]

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
}}
"""
   return out

depth_cluster_system_instruction = """You are an clusterer that is performing taxonomy depth expansion, which is defined as adding subcategory nodes deeper to a given root_topic node, these being children concepts/topics which EXCLUSIVELY fall under the specified parent node and not the parent\'s siblings. For example, given a taxonomy of NLP tasks, expanding "text_classification" depth-wise (where its siblings are [\"named_entity_recognition\", \"machine_translation\", and \"question_answering\"]) would create the children nodes, [\"sentiment_analysis\", \"spam_detection\", and \"document_classification\"] (any suitable number of children). On the other hand, \"open_domain_question_answering\" SHOULD NOT be added as it belongs to sibling, \"question_answering\".

You are choosing your new subtopic clusters based on which subtopics of the parent topic are covered by papers that discuss the parent node. Your job is to identify unique clusters formed from the input set of paper topics. For each cluster you identify, you must provide a cluster name (in similar format to the paper_topics) as its key, a 1 sentence description of the cluster name, and a list of all the input paper_topics covered within the cluster. MAKE SURE EVERY NEW SUBTOPIC IS DISTINCT AND HAS THE SAME LEVEL OF GRANULARITY/SPECIFICITY. Also make sure that each of your new topic clusters are UNIQUE; they SHOULD NOT currently exist within the existing set of nodes (existing_nodes)."""

class DepthClusterSchema(BaseModel):
    label: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=250)]
    covered_paper_topics: conlist(str, min_length=1, max_length=20)

class DepthClusterListSchema(BaseModel):
    new_cluster_topics: conlist(DepthClusterSchema, min_length=1, max_length=10)



def depth_cluster_main_prompt(options, node, ancestors, all_node_labels):
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

Your new cluster topics SHOULD NOT BE ANY OF THE FOLLOWING:
<existing_nodes>
{all_node_labels}
</existing_nodes>

<paper_topics>
Below is a dictionary of paper topics, where each key is the candidate node label and value is number of papers which are mapped to that candidate node:
candidate_node_labels:\n{str(options)}
</paper_topics>

</input>

What are the primary sub-{node.dimension} topic clusters under the parent_node topic, {node.label}, that would best encompass the above <paper_topics>?
These should be non-overlapping topic clusters that best represent and partition all of paper_topics (maximize the number of papers that are mapped to each). They should all be siblings (same level of depth/specificity) under the parent_node within the taxonomy. Each new cluster topic that you suggest should be a more specific subtopic under the parent_node, {node.label}, and be a type of {node.dimension}. However, they should all be equally unique (non-duplicates) and no single paper should be able to fall into both clusters easily.\n

Your output should be in the following JSON format with a minimum of one subtopic cluster and a maximum of five:
{{
  "new_cluster_topics":
  [
    {{
      "label": <string sub-{node.dimension} label at the deeper level of depth/specificity as the parent_node, {node.label}>,
      "description": <string sub-{node.dimension} sentence-long description>,
      "covered_paper_topics": <list of all the input paper_topics covered within this sub-{node.dimension}>
    }},
    ...
  ]
}}
---

Your output JSON:

"""
  return out