init_enrich_prompt = "You are a helpful assistant that performs taxonomy enrichment."

main_enrich_prompt = lambda dict_str: f"""I am providing you a JSON which contains a taxonomy detailing concepts for "using llms on graphs" in NLP research papers. Each JSON key within the "children" dictionary represents a taxonomy concept node. Can you fill in the "example_key_phrases" and "example_sentences" fields for each concept node (enrichment of both the root node and its children/descendants) that contains these fields? The required fields are already present for you, so you do not need to create any new keys for concepts without them. Here are the instructions for each field under concept A:

1. "example_key_phrases": This is a list (Python-formatted) of 20 key phrases commonly used amongst NLP research papers that exclusively discuss that concept node (concept A's key phrases should be highly relevant to its concept A's parent concept, and not or rarely be mentioned in one of its sibling concepts B; A and B share the same parent concept). All added key phrases should be 1-3 words, lowercase, and have spaces replaced with underscores (e.g., "key_phrase"). Each key phrase should be unique.
2. "example_sentences": This is a list (Python-formatted) of 10 key sentences that could be used to discuss the concept node A within an NLP research paper. These key sentences should be specific, not generic, to Concept A (also relevant to its parents or ancestors), and unable to be used to describe any other sibling concepts.

Given the taxonomy JSON below, output your enriched taxonomy JSON (only output your final JSON, no other information) following the above rules and taxonomy rules in general. Your entire response/output is going to consist of a single JSON object \{\}, and you will NOT wrap it within JSON Markdown markers.
---
{dict_str}
---

JSON Output:
---
"""