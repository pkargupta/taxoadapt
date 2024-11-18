from model_definitions import constructPrompt, promptLLM
from prompts import CommonSenseSchema
import json
from utils import clean_json_string

init_enrich_prompt = "You are a helpful assistant that performs taxonomy enrichment using realistic specific keywords and sentences that would be used in NLP research papers. These realistic keywords and sentences will be used to identify research papers which discuss a specific taxonomy node."

main_simple_enrich_prompt = lambda node, ancestors, sibs: f'''"{node.label}" is a subtopic of {ancestors}. Please generate realistic key terms and sentences about the '{node.label}' topic that are relevant to '{node.label}' but irrelevant to the topics: {sibs}. The terms should be short (1-3 words), concise, and distinctive to {node.label}. The sentences should have specific details and resemble realistic sentences found in NLP research papers.

Here is more information about the node that may help you in identifying key terms and sentences:
Description of {node.label}: {node.description}
Types of datasets used for {node.label}: {node.datasets}
Types of methodologies used for {node.label}: {node.methodologies}
Types of evaluation methods/metrics used for {node.label}: {node.evaluation_methods}
Types of applications that {node.label} is used for: {node.applications}

Your output format should be in the following JSON format (where node_to_enrich, id and description for {node.label} should match their respective values in the input taxonomy 'input_taxo' aka COPIED OVER FROM input_taxo):
---
{{
    "node_to_enrich: "{node.label}"
    "id": "{node.id}",
    "example_key_phrases": <list of 20 diverse, short terms where values are realistic and relevant to '{node.label}' and DISSIMILAR to any of the following: '{sibs}'>,
    "example_sentences": <list of 10 diverse sentences where values are sentences used in papers about '{node.label}' and DISSIMILAR to any of the following: '{sibs}'>
}}
---
'''

def enrich_node(args, node, ancestors):
    prompts = []
    sibs = [i.label for i in node.get_siblings()]
    prompts.append(constructPrompt(init_enrich_prompt, main_simple_enrich_prompt(node, ancestors, sibs)))

    output = promptLLM(args, prompts, schema=CommonSenseSchema, max_new_tokens=2000)
    output_dict = [json.loads(clean_json_string(c)) if "```" in c else json.loads(c.strip()) for c in output]

    node.phrases = output_dict['example_key_phrases']
    node.sentences = output_dict['example_sentences']