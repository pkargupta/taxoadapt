import os
from taxonomy import Taxonomy, Paper
from utils import filter_phrases
import subprocess
import shutil
import argparse
import re

def main(args):
    print("########### READING IN PAPERS ###########")
    collection = []
    id = 0
    with open(args.input_file, "r") as f:
        papers = f.read().strip().splitlines()
        for p in papers:
            title = re.findall(r'title\s*:\s*(.*) ; ', p, re.IGNORECASE)
            abstract = re.findall(r'abstract\s*:\s*(.*)', p, re.IGNORECASE)
            collection.append(Paper(id, title, abstract))
            id += 1

    # input: track, dimension -> get base taxonomy (2 levels) -> Class Tree, Class Node (description, seed words)
    print("########### BASE TAXONOMY ###########")
    taxo = Taxonomy(args.track, args.dim)
    base_taxo = taxo.buildBaseTaxo(levels=1)

    print(base_taxo)

    # format the input keywords file for seetopic -> get phrases -> filter using LLM
    dir_name = (args.track + "_" + args.dim).lower().replace(" ", "_")

    if not os.path.exists(f"SeeTopic/{dir_name}"):
        os.makedirs(f"SeeTopic/{dir_name}")

    if not os.path.exists(f"SeeTopic/{dir_name}/{dir_name}.txt"):
        shutil.copyfile(args.input_file, f"SeeTopic/{dir_name}/{dir_name}.txt")
    
    ## get first level of children
    print("########### PHRASE MINING FOR LEVEL 1 ###########")
    children_with_terms = taxo.root.getChildren(terms=True)
    with open(f"SeeTopic/{dir_name}/keywords_0.txt", "w") as f:
        for idx, c in enumerate(children_with_terms):
            str_c = ",".join(c[1])
            f.write(f"{idx}:{c[0]},{str_c}")
    
    os.chdir("./SeeTopic")
    subprocess.check_call(['./seetopic.sh', dir_name, str(args.iters), "bert_full_ft"])
    os.chdir("../")

    with open(f"./SeeTopic/{dir_name}/keywords_seetopic.txt", "r") as f:
        children_phrases = [i.strip().split(":")[1].split(",") for i in f.readlines()]
        filtered_children_phrases = []
        for c_id, c in enumerate(taxo.root.children):
            # filter the child phrases
            child_phrases = filter_phrases(c, f"{c}: {children_phrases[c_id]}\n")
            filtered_children_phrases.append(child_phrases)

    for c_id, c in enumerate(taxo.root.children):
        c.addTerms(filtered_children_phrases[c_id], addToParent=True)
    

    # (initial relevant pool) identify papers which contain exact-matched terms -> class Paper (relevant segments, sentences, phrases)
    print("########### INITIAL RELEVANT POOL OF PAPERS ###########")
    




    # (primary focus pool) identify papers which propose methods involving such topics → utilize class-oriented sentence representations

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--track', type=str, default='Text Classification')
    parser.add_argument('--dim', type=str, default='Methodology')
    parser.add_argument('--input_file', type=str, default='datasets/sample_1k.txt')
    parser.add_argument('--iters', type=int, default=4)

    args = parser.parse_args()
    main(args)