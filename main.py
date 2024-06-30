import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"
os.environ['HF_HOME'] = '/shared/data3/pk36/.cache'
from taxonomy import Taxonomy, Paper
from utils import filter_phrases
import subprocess
import shutil
import pickle as pk
import numpy as np
import argparse
from model_definitions import sentence_model


def main(args):

    # input: track, dimension -> get base taxonomy (2 levels) -> Class Tree, Class Node (description, seed words)
    print("########### READING IN PAPERS & CONSTRUCTING BASE TAXONOMY ###########")
    taxo = Taxonomy(args.track, args.dim, args.input_file)
    base_taxo = taxo.buildBaseTaxo(levels=1, num_terms=20)

    print(base_taxo)

    # format the input keywords file for seetopic -> get phrases -> filter using LLM
    dir_name = (args.track + "_" + args.dim).lower().replace(" ", "_")

    if not os.path.exists(f"SeeTopic/{dir_name}"):
        os.makedirs(f"SeeTopic/{dir_name}")

    if not os.path.exists(f"SeeTopic/{dir_name}/{dir_name}.txt"):
        shutil.copyfile(args.input_file, f"SeeTopic/{dir_name}/{dir_name}.txt")

    ## get first level of children
    children_with_terms = taxo.root.getChildren(terms=True)
    with open(f"SeeTopic/{dir_name}/keywords_0.txt", "w") as f:
        for idx, c in enumerate(children_with_terms):
            str_c = ",".join(c[1])
            f.write(f"{idx}:{c[0]},{str_c}\n")
    
    os.chdir("./SeeTopic")
    subprocess.check_call(['./seetopic.sh', dir_name, str(args.iters), args.model])
    os.chdir("../")

    # read in raw and static embs

    raw_emb = {}
    with open(f'./SeeTopic/{dir_name}/embedding_{args.model}.txt') as fin:
        for line in fin:
            data = line.strip().split()
            if len(data) != 769:
                continue
            word = data[0]
            emb = np.array([float(x) for x in data[1:]])
            emb = emb / np.linalg.norm(emb)
            raw_emb[word] = emb

    taxo.raw_emb = raw_emb

    if os.path.exists(os.path.join('SeeTopic/text_classification_methodology/static_emb.pk')):
        with open(os.path.join('SeeTopic/text_classification_methodology/static_emb.pk'), "rb") as f:
            static_emb = pk.load(f)
    taxo.static_emb = static_emb

    with open(f"./SeeTopic/{dir_name}/keywords_seetopic.txt", "r") as f:
        children_phrases = [i.strip().split(":")[1].split(",") for i in f.readlines()]
        filtered_children_phrases = []
        for c_id, c in enumerate(taxo.root.children):
            # other parents
            other_parents = "\n".join([f"{i.label} -> {i.desc}" for i in taxo.root.children if i != c])
            # filter the child phrases
            child_phrases = filter_phrases(c, f"{c}: {children_phrases[c_id]}\n", word2emb, other_parents=other_parents)
            filtered_children_phrases.append(child_phrases)

    for c_id, c in enumerate(taxo.root.children):
        c.addTerms(filtered_children_phrases[c_id], mined=True, addToParent=True)
    

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
    parser.add_argument('--model', type=str, default="bert_full_ft")

    args = parser.parse_args()
    main(args)