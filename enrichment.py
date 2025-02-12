from model_definitions import constructPrompt

init_enrich_prompt = "You are a helpful assistant that performs taxonomy enrichment using realistic specific keywords and sentences that would be used in research papers. These realistic keywords and sentences will be used to identify research papers which discuss a specific taxonomy node."

main_simple_enrich_prompt = lambda node, ancestors, sibs: f'''"{node.label}" (a type of {node.dimension}) is a subcategory of all of the following categories: {ancestors}. Please generate realistic key terms and sentences about the '{node.label}' topic that are relevant to '{node.label}' but irrelevant to the topics: {sibs}. The terms should be short (1-3 words), concise, and distinctive to {node.label}. The sentences should have specific details and resemble realistic sentences found in research papers.

Here is more information about the node that may help you in identifying key terms and sentences:
Description of {node.label}: {node.description}
{node.label}'s category type: {node.dimension}

Your output format should be in the following JSON format (where node_to_enrich, id and description for {node.label} should match their respective values in the input taxonomy 'input_taxo' aka COPIED OVER FROM input_taxo):
---
{{
    "node_to_enrich: "{node.label}"
    "id": "{node.id}",
    "commonsense_key_phrases": <list of 20 diverse, short terms where values are realistic and relevant to '{node.label}' and DISSIMILAR to any of the following: '{sibs}'>,
    "commonsense_sentences": <list of 10 diverse, longer sentences where values are realistic, specific sentences used in papers about '{node.label}' and DISSIMILAR to any of the following: '{sibs}'>
}}
---

Output only your JSON:
'''

def enrich_node_prompt(args, node, ancestors):
    sibs = [i.label for i in node.get_siblings()]
    prompt = constructPrompt(args, init_enrich_prompt, main_simple_enrich_prompt(node, ancestors, sibs))

    return prompt