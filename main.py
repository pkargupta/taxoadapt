import os
os.environ['HF_HOME'] = '/shared/data3/pk36/.cache'
from taxonomy import Node, Taxonomy
import subprocess
import pickle as pk
from tqdm import tqdm
import numpy as np
from collections import deque
import json
import time
import argparse
from model_definitions import llama_8b_model, promptLlama, constructPrompt
from utils import *
from prompts import *

def commonSenseEnrich(root_node, taxo_dict, batch=True):
    # phrase and sentence-level enrichment

    ## constructing prompts
    if batch:
        prompts = []
        for child in root_node.children:
            temp_dict = taxo_dict.copy()
            temp_dict[root_node.label]['children'] = {c.label:({'description':c.description} if c.node_id != child.node_id else taxo_dict[root_node.label]['children'][child.label]) for c in root_node.children}

            prompts.append(constructPrompt(init_enrich_prompt, main_enrich_prompt(temp_dict)))
    else:
        prompts = constructPrompt(init_enrich_prompt, main_enrich_prompt(taxo_dict))

    ## generation
    output = promptLlama(prompts, max_new_tokens=5000)
    if batch:
        output_dict = [json.loads(clean_json_string(c)) if "```json" in c else json.loads(c.strip()) for c in output]
    else:
        output_dict = [json.loads(clean_json_string(output) if "```json" in output else output.strip())]

    ## merging all dictionaries
    print("merging all enrichment dictionaries...")

    for c in tqdm(output_dict):

        queue = deque(c)

        while queue:
            current_dict = queue.popleft()
            current_node = root_node.findChild(current_dict['id'])
            if current_node is not None:
                updateEnrichment(current_node, current_dict['example_key_phrases'], current_dict['example_sentences'])

                for child_label, child_dict in current_dict['children'].items():
                    queue.append(child_dicts)
    return

def main(args):

    start = time.time()

    print("########### READING INPUT ###########")

    # create taxonomy from input
    root, id2label, label2id = createGraph(os.path.join(args.data_dir), 'labels_with_desc.txt')

    taxo = Taxonomy(root)

    taxo_dict = taxo.toDict(cur_node=taxo.root)
    with open(f'datasets/{args.dataset}/initial.json', 'w') as fp:
        json.dump(taxo_dict, fp, indent=4)

    # first do common sense phrase, sentence enrichment on taxonomy nodes
    enrich_start = time.time()
    commonSenseEnrich(taxo.root, taxo_dict, True)
    enrich_end = time.time()
    print(f"Time taken: {(enrich_end - enrich_start)/60} minutes")

    # add common-sense phrases to AutoPhrase
    with open("preprocessing/AutoPhrase/data/EN/wiki_quality_orig.txt", "r") as f:
        all_phrases = [w.strip() for w in f.readlines()]
        for a in aspects:
            all_phrases.append(a.replace("_", " "))
        all_phrases.extend([w.replace("_", " ") for a in keywords for w in a])

    with open("preprocessing/AutoPhrase/data/EN/wiki_quality.txt", "w") as f:
        for w_id, w in enumerate(all_phrases):
            if w_id == (len(all_phrases) - 1):
                f.write(f"{w}")
            else:
                f.write(f"{w}\n")



    print("########### PRE-PROCESSING BOTH CORPORA ###########")
    if args.override or (not os.path.exists(f"datasets/{args.dataset}/phrase_{args.dataset}.txt")):
        # pre-process
        os.chdir("./preprocessing")
        subprocess.check_call(['./auto_phrase.sh', args.dataset])
        os.chdir("../")

    else:
        print("already pre-processed!")


    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='llm_graph')
    parser.add_argument('--iters', type=int, default=4)
    parser.add_argument('--model', type=str, default="bert_full_ft")
    parser.add_argument('--override', type=bool, default=True)
    parser.add_argument('--max_depth', type=int, default=5)

    args = parser.parse_args()

    # inputs: external knowledge corpus & specific corpus
    args.data_dir = f"datasets/{args.dataset}/"
    args.input_file = f"datasets/{args.dataset}/phrase_{args.dataset}.txt"


    main(args)