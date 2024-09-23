from pydantic import BaseModel, conset, StringConstraints, Field
from typing_extensions import Annotated

def baseline_prompt(paper, node):
   
   cats = "\n".join([f"{node.description}" for c in node.children])

   return f'''You will be provided with a research paper title and abstract. Please select the categories that this paper should be placed under. We provide the list of categories and their respective descriptions. Just give the category names as shown in the list.

title: {paper.title}
abstract: {paper.abstract}

categories:
{cats}
'''

init_enrich_prompt = "You are a helpful assistant that performs taxonomy enrichment using realistic specific keywords and sentences that would be used in NLP research papers. These realistic keywords and sentences will be used to identify papers research papers which discuss a specific taxonomy node."

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

main_enrich_prompt = lambda node, sibs, dict_str: f'''I am providing you a JSON which contains a taxonomy detailing concepts in NLP research papers (tag 'input_taxo'). Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases" and "example_sentences" fields for the specified node (tag 'node_to_enrich')? A research paper relevant to 'node_to_enrich' will be relevant to all concept nodes present in the taxonomy path to the node, 'node_to_enrich', as listed in 'path_to_node'. Here are your instructions on how to enrich the fields for node, 'node_to_enrich':

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