import os
from taxonomy import Taxonomy, Node


# input: track, dimension -> get base taxonomy (2 levels) -> Class Tree, Class Node (description, seed words)



# format the input keywords file for seetopic -> get phrases -> filter using LLM

# (initial relevant pool) identify papers which contain exact-matched terms -> class Paper (relevant segments, sentences, phrases)

# (primary focus pool) identify papers which propose methods involving such topics

# fine-tune [to-be hierarchical] entailment model

# (paper-based enrichment) if all papers are mapped to all level-nodes, no enrichment
#                          elif node is high density + some neutral papers --> either abandon papers or gather more context (retrieval)
#                          elif node is high density + many contraditions, unmapped papers --> enrich siblings (open question)
#                                    enrich siblings: use LLM to identify clusters within unmapped papers -> mine terms
#                          elif node is high density --> enrich children (common-sense)
#                          else node is low density --> prune

# iterative feedback: construct hierarchical entailment hypotheses for further enrichment