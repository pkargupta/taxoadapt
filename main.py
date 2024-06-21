import os
from taxonomy import Taxonomy
import subprocess
import shutil


def main():
    # input: track, dimension -> get base taxonomy (2 levels) -> Class Tree, Class Node (description, seed words)
    track = "Text Classification"
    dim = "Methodology"
    input_papers = "datasets/sample_1k.txt"

    print("########### BASE TAXONOMY ###########")
    taxo = Taxonomy(track, dim)
    base_taxo = taxo.buildBaseTaxo(levels=1)

    print(base_taxo)

    # format the input keywords file for seetopic -> get phrases -> filter using LLM
    dir_name = (track + dim).lower().replace(" ", "_")

    if not os.path.exists(f"SeeTopic/{dir_name}"):
        os.makedirs(f"SeeTopic/{dir_name}")
    
    shutil.copyfile(input_papers, f"SeeTopic/{dir_name}/{dir_name}.txt")
    
    ## get first level of children
    children_with_terms = taxo.root.getChildren(terms=True)
    with open(f"SeeTopic/{dir_name}/keywords_0.txt", "w") as f:
        for idx, c in enumerate(children_with_terms):
            str_c = ",".join(c[1])
            f.write(f"{idx}:{c[0]},{str_c}")
    
    subprocess.check_call(['./SeeTopic/seetopic.sh', 'arg1', 'arg2', arg3])


    # (initial relevant pool) identify papers which contain exact-matched terms -> class Paper (relevant segments, sentences, phrases)

    # (primary focus pool) identify papers which propose methods involving such topics

    # fine-tune [to-be hierarchical] entailment model

    # (paper-based enrichment) if all papers are mapped to all level-nodes, no enrichment
    #                          elif node is high density + some neutral papers --> either abandon papers or gather more context (retrieval)
    #                          elif node is high density + many contradictions, unmapped papers --> enrich siblings (open question)
    #                                    enrich siblings: use LLM to identify clusters within unmapped papers -> mine terms
    #                          elif node is high density --> enrich children (common-sense)
    #                          else node is low density --> prune

    # iterative feedback: construct hierarchical entailment hypotheses for further enrichment
    return

if __name__ == "__main__":
    main()