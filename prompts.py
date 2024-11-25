from pydantic import BaseModel, conset, StringConstraints, Field
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
  
def multi_dim_prompt(node):
    topic = node.label
    ancestors = ", ".join([ancestor.label for ancestor in node.get_ancestors()])

    system_instruction = f'You are a helpful assistant that constructs taxonomies for a given root topic: {topic}{"" if ancestors == "" else " (relevant to" + ancestors + ")"}. Keep in mind that research papers will be mapped to the nodes within your constructed taxonomy.'

    main_prompt = f'Your root_topic is: {topic}\nA subcategory is a specific division within a broader category that organizes related items or concepts more precisely. Output up to 5 child, subcategories of {node.dimension} that fall under {topic} and generate corresponding sentence-long descriptions for each. Make sure each type is unique to the topics: {topic}, {ancestors}.'

    if ('domain' in node.dimension) or ('application' in node.dimension):
        main_prompt += f'\n Remember that {node.dimension} means a real-world domain category in which an NLP paper can be applied to (for example, news or science could be a subcategory of {node.dimension}).'

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




def baseline_prompt(paper, node):
   
   cats = "\n".join([f"{node.description}" for c in node.children])

   return f'''You will be provided with a research paper title and abstract. Please select the categories that this paper should be placed under. We provide the list of categories and their respective descriptions. Just give the category names as shown in the list.

title: {paper.title}
abstract: {paper.abstract}

categories:
{cats}
'''

init_enrich_prompt = "You are a helpful assistant that performs taxonomy enrichment using realistic specific keywords and sentences that would be used in NLP research papers. These realistic keywords and sentences will be used to identify research papers which discuss a specific taxonomy node."

init_classify_prompt = "You are a helpful assistant that identifies the class labels for the provided NLP research paper, performing multi-label classification."

bulk_enrich_prompt = lambda dict_str: f'''I am providing you a JSON which contains a taxonomy detailing concepts for NLP research papers. Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases" and "example_sentences" fields for each concept node (enrichment of both the root node and its children/descendants) that contains these fields? The required fields are already present for you, so you do not need to create any new keys for concepts without them. Here are the instructions for each field under concept A:

1. "example_key_phrases": This is a list (Python-formatted) of 20 key phrases (e.g., SUBTOPICS of the given concept node A) commonly used amongst NLP research papers that EXCLUSIVELY DISCUSS that concept node (concept A's key phrases/subtopics should be highly relevant to its concept A's parent concept, and NOT OR RARELY be mentioned in ANY of its SIBLING concepts B; A and B share the same parent concept). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., "key_phrase"). Each key phrase should be unique.
2. "example_sentences": This is a list (Python-formatted) of 10 key sentences that could be used to discuss the concept node A within an NLP research paper. These key sentences should be SPECIFIC, not generic, to Concept A (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts. Utilize your knowledge of the concept node A (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.

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

main_simple_enrich_prompt = lambda taxo, node, sibs: f'''"{node.label}" is a topic in Natural Language Processing (NLP){parent_prompt(taxo, node)}. Please generate 20 realistic key terms and sentences about the '{node.label}' topic that are relevant to '{node.label}' but irrelevant to the topics: {sibs}. The terms should be short (1-3 words), concise, and distinctive to {node.label}. The sentences should have specific details and resemble realistic sentences found in NLP research papers.

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

main_long_enrich_prompt = lambda node, sibs, dict_str: f'''I am providing you a JSON which contains a taxonomy detailing concepts in NLP research papers (tag 'input_taxo'). Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases" and "example_sentences" fields for the specified node (tag 'node_to_enrich')? A research paper relevant to 'node_to_enrich' will be relevant to all concept nodes present in the taxonomy path to the node, 'node_to_enrich', as listed in 'path_to_node'. Here are your instructions on how to enrich the fields for node, 'node_to_enrich':

1. "example_key_phrases": This is a list (Python-formatted) of 20 key, realistic phrases (e.g., SUBTOPICS of the given 'node_to_enrich') commonly written within NLP research papers that EXCLUSIVELY DISCUSS 'node_to_enrich'. 'node_to_enrich's key phrases/subtopics should be highly relevant to all of 'node_to_enrich's ancestors listed in 'path_to_node', and NOT be relevant to ANY other non-ancestor or siblings of 'node_to_enrich' (siblings of 'node_to_enrich' are specified in tag 'siblings' below). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., "key_phrase"). Each key phrase should be unique.
2. "example_sentences": This is a list (Python-formatted) of 10 key, realistic sentences that could be written in an NLP research paper to discuss 'node_to_enrich'. These key sentences should be SPECIFIC, not generic, to 'node_to_enrich' (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts (tag 'siblings'). Utilize your knowledge of the 'node_to_enrich' (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.

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

main_classify_prompt = lambda node, paper: f'''Given the 'title', 'abstract', and 'content' (provided below) of an NLP research paper that uses large language models for graphs, select the class labels (tag 'class_options') that should be assigned to this paper (multi-label classification). If the research paper SHOULD NOT be labeled with any of the classes in 'class_options', then output an empty list. We provide additional descriptions (tag 'class_descriptions') for each class option for your reference.

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

main_enrich_prompt_paper = lambda node, dict_str: f'''I am providing you a JSON which contains a taxonomy detailing concepts for "using llms on graphs" in NLP research papers (tag 'input_taxo'). Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases", "example_sentences", "example_paper_titles", and "example_paper_abstracts" fields for the specified node (tag 'node_to_enrich')? A research paper relevant to 'node_to_enrich' will be relevant to all concept nodes present in the path to 'node_to_enrich', as listed in 'path_to_node'. Here are your instructions on how to enrich the fields for node, 'node_to_enrich':

1. "example_key_phrases": This is a list (Python-formatted) of 20 key, realistic phrases (e.g., SUBTOPICS of the given concept node A) commonly written within NLP research papers that EXCLUSIVELY DISCUSS that concept node (node A's key phrases/subtopics should be highly relevant to its node A's parent concept, and NOT be relevant to ANY other non-ancestor or descendants of node A; A and B share the same parent concept). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., "key_phrase"). Each key phrase should be unique.
2. "example_sentences": This is a list (Python-formatted) of 10 key, realistic sentences that could be written in an NLP research paper to discuss the concept node A. These key sentences should be SPECIFIC, not generic, to node A (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts. Utilize your knowledge of the concept node A (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.
3. "example_paper_titles": This is a list (Python-formatted) of 2 realistic, detailed titles of an NLP research paper that discusses 'node_to_enrich', as well as all nodes in 'path_to_node'. Each title should not be able to be placed under any other node within 'input_taxo' that is not an ancestor or descendant of 'node_to_enrich'. Each title should correspond to the abstract with the same index in "example_paper_abstracts".
4. "example_paper_abstracts": This is a list (Python-formatted) of 2 realistic, detailed NLP research paper abstracts (paragraph long) that discusses 'node_to_enrich', as well as all nodes in 'path_to_node'. Each abstract should not be able to be placed under any other node within 'input_taxo' that is not an ancestor or descendant of 'node_to_enrich'. Each abstract should correspond to the title with the same index in "example_paper_titles".

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
    "example_paper_titles": <list of strings where values are realstic NLP research paper titles that are relevant to 'node_to_enrich' and 'path_to_node'>
    "example_paper_abstracts": <list of strings where values are realstic NLP research paper abstracts that are relevant to 'node_to_enrich' and 'path_to_node'>
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
  "title": "Taxonomy NLP Research Concept",
  "description": "An NLP research concept present within the input taxonomy, input_taxo",
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
      "description": "20 key phrases (e.g., SUBTOPICS of the given concept node A) commonly used amongst NLP research papers that EXCLUSIVELY DISCUSS that concept node (node A's key phrases/subtopics should be highly relevant to its node A's parent concept, and NOT be relevant to ANY siblings of node A; node A and sibling B share the same parent concept). All added key phrases/subtopics should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., 'key_phrase'). Each key phrase should be unique.",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": True
    },
    "example_sentences": {
      "description": "10 key sentences that could be used to discuss the concept node A within an NLP research paper. These key sentences should be SPECIFIC, not generic, to node A (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts. Utilize your knowledge of the concept node A (including the provided corresponding 'description' of node A and its ancestors/parent node) to form your example sentences.",
      "type": "array",
      "items": {
        "type": "string"
      },
      "minItems": 1,
      "uniqueItems": True
    }
  }
}