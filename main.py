import os
os.environ['HF_HOME'] = '/shared/data3/pk36/.cache'
import subprocess
import pickle as pk
from tqdm import tqdm
import numpy as np
from collections import deque
import json
import time
import argparse
from scipy import stats
from itertools import compress
from taxonomy import Node, Taxonomy
from model_definitions import sentence_model, promptLlamaVLLM, constructPrompt, promptLlamaSamba, promptGPT
from utils import *
from prompts import *

def commonSenseEnrich(root_node, dict_str, batch=True):
    # phrase and sentence-level enrichment

    # construct prompts for each node
    prompts = []
    queue = deque([root_node])
    while queue:
        current_node = queue.popleft()
        sibs = [i.label for i in current_node.parents[0].children if i != current_node]
        prompts.append(constructPrompt(init_enrich_prompt, main_enrich_prompt(current_node, sibs, dict_str), api=False))

        for child in current_node.children:
            queue.append(child)

    output = promptLlamaVLLM(prompts, schema=CommonSenseSchema, max_new_tokens=2000)
    try:
        if batch:
            output_dict = [json.loads(clean_json_string(c)) if "```json" in c else json.loads(c.strip()) for c in output]
        else:
            output_dict = [json.loads(clean_json_string(output) if "```json" in output else output.strip())]
    except:
        return prompts, output, None, None

    ## merging all dictionaries
    print("merging all enrichment dictionaries...")

    common_sense_phrases = []
    common_sense_sentences = []

    for c in tqdm(output_dict):
        c['example_key_phrases'] = [p.lower().replace(' ', '_') for p in c['example_key_phrases']]

        current_node = root_node.findChild(c['id'])
        if current_node is not None:
            common_sense_phrases.extend(c['example_key_phrases'])
            common_sense_sentences.append(c['description'])
            common_sense_sentences.extend(c['example_sentences'])
            updateEnrichment(current_node, c['example_key_phrases'], c['example_sentences'], c['description'])

    return prompts, output_dict, list(set(common_sense_phrases)), list(set(common_sense_sentences))

def classAnnotate(root_node, papers, batch=True):
    # phrase and sentence-level enrichment

    # construct prompts for each node
    prompts = []
    queue = deque([root_node])
    while queue:
        current_node = queue.popleft()

        if len(current_node.children) > 0:
            for paper in papers:
                prompts.append(constructPrompt(init_classify_prompt, main_classify_prompt(current_node, paper)))

            for child in current_node.children:
                queue.append(child)

    output = promptLlamaVLLM(prompts, schema=ClassifySchema, max_new_tokens=5000)
    try:
        if batch:
            output_dict = [json.loads(clean_json_string(c)) if "```json" in c else json.loads(c.strip()) for c in output]
        else:
            output_dict = [json.loads(clean_json_string(output) if "```json" in output else output.strip())]
        
        return prompts, output, output_dict
    except:
        return prompts, output, None
    

def rankPhrases(phrase_pool, curr_node, taxo, use_class_emb=False, granularity='phrases', out_phrases=False):
    if use_class_emb:
        embs, sim_diff, keep_phrase = compareClassesEmbs(phrase_pool, taxo, curr_node, granularity)
    else:
        embs, sim_diff, keep_phrase = compareClasses(phrase_pool, taxo, curr_node, granularity)
    
    filtered_embs = embs[keep_phrase, :]
    filtered_class_phrases = list(compress(phrase_pool, keep_phrase))
    filtered_diffs = sim_diff[keep_phrase]

    # print(f'{curr_node.label}: {len(class_phrases) - len(filtered_class_phrases)} {granularity} filtered!')
    
    phrase2emb = {p:e for p, e in zip(filtered_class_phrases, filtered_embs)}

    ranks = {i: r for r, i in enumerate(np.argsort(-np.array(filtered_diffs)))}
    ranked_tok = {filtered_class_phrases[idx]:rank for idx, rank in ranks.items()}
    ranked_phrases = list(ranked_tok.keys())
    curr_node.all[granularity] = ranked_phrases
    
    class_emb = average_with_harmonic_series(np.array([phrase2emb[p] for p in ranked_phrases]))
    # curr_node.emb[granularity] = class_emb

    if out_phrases:
        return ranked_phrases, class_emb
    else:
        return class_emb

def computeClassEmb(curr_node, taxo, class_emb=False, granularity='phrases', out_phrases=False):
    if granularity == 'mixed':
        class_phrases = curr_node.getAllTerms(granularity='phrases') + curr_node.getAllTerms(granularity='sentences')
    else:
        class_phrases = curr_node.getAllTerms(children=False, granularity=granularity)
    
    # class_phrases = [curr_node.label] if granularity == 'phrases' else [curr_node.description]
    
    if class_emb:
        embs, sim_diff, keep_phrase = compareClassesEmbs(class_phrases, taxo, curr_node, granularity, parent_weight=0.2)
    else:
        embs, sim_diff, keep_phrase = compareClasses(class_phrases, taxo, curr_node, granularity)
    
    filtered_embs = embs[keep_phrase, :]
    filtered_class_phrases = list(compress(class_phrases, keep_phrase))
    filtered_diffs = sim_diff[keep_phrase]

    # print(f'{curr_node.label}: {len(class_phrases) - len(filtered_class_phrases)} {granularity} filtered!')
    
    phrase2emb = {p:e for p, e in zip(filtered_class_phrases, filtered_embs)}

    ranks = {i: r for r, i in enumerate(np.argsort(-np.array(filtered_diffs)))}
    ranked_tok = {filtered_class_phrases[idx]:rank for idx, rank in ranks.items()}
    ranked_phrases = list(ranked_tok.keys())
    curr_node.all[granularity] = ranked_phrases
    
    class_emb = average_with_harmonic_series(np.array([phrase2emb[p] for p in ranked_phrases]))
    curr_node.emb[granularity] = class_emb

    if out_phrases:
        return ranked_phrases, class_emb
    else:
        return class_emb

def main(args):

    start = time.time()

    print("########### CONSTRUCTING GRAPH & COMMON-SENSE ENRICHMENT ###########")

    # create taxonomy from input
    root, id2label, label2id = createGraph(os.path.join(args.data_dir), 'labels_with_desc.txt')

    taxo = Taxonomy(root)

    taxo_dict = taxo.toDict(cur_node=taxo.root)
    with open(f'datasets/{args.dataset}/initial.json', 'w') as fp:
        json.dump(taxo_dict, fp, indent=4)
    
    dict_str = json.dumps(taxo_dict, indent=4)

    # first do common sense phrase, sentence enrichment on taxonomy nodes
    enrich_start = time.time()
    all_common_phrases = commonSenseEnrich(taxo.root, dict_str, True)
    enrich_end = time.time()
    print(f"Time taken: {enrich_end - enrich_start} seconds ({(enrich_end - enrich_start)/60} minutes)")

    updated_dict = taxo.toDict(cur_node=taxo.root)
    with open(f'datasets/{args.dataset}/enriched.json', 'w') as fp:
        json.dump(updated_dict, fp, indent=4)


    print("########### PRE-PROCESSING BOTH CORPORA ###########")

    # add common-sense phrases to AutoPhrase
    with open("preprocessing/AutoPhrase/data/EN/wiki_quality_orig.txt", "r") as f:
        all_phrases = [w.strip() for w in f.readlines()]
        for a in all_common_phrases:
            all_phrases.append(a.strip().replace("_", " "))

    with open("preprocessing/AutoPhrase/data/EN/wiki_quality.txt", "w") as f:
        for w_id, w in enumerate(all_phrases):
            if w_id == (len(all_phrases) - 1):
                f.write(f"{w}")
            else:
                f.write(f"{w}\n")

    
    if args.override:
        # pre-process
        os.chdir("./preprocessing")
        subprocess.check_call(['./auto_phrase.sh', args.dataset, args.internal])
        subprocess.check_call(['./auto_phrase.sh', args.dataset, args.external])
        os.chdir("../")

    else:
        print("already pre-processed!")


    collection = taxo.createCollection(os.path.join(args.data_dir, "phrase_" + args.internal), os.path.join(args.data_dir, args.groundtruth), external=False)
    external_collection = taxo.createCollection(os.path.join(args.data_dir, "phrase_" + args.external), external=True)



    print("########### EXTERNAL + INTERNAL ENRICHMENT ###########")

    target_node = taxo.root
    
    class_ids = [child.node_id for child in target_node.children]
    phrase_class_embs = [computeClassEmb(curr_node, taxo, granularity='phrases') for curr_node in target_node.children]
    sent_class_embs = [computeClassEmb(curr_node, taxo, granularity='sentences') for curr_node in target_node.children]
    joint_class_embs = [(p+s)/2 for p, s in zip(phrase_class_embs, sent_class_embs)]

    labels = []
    scores = []
    class_map = {i:[] for i in class_ids}

    for p_id, p in tqdm(target_node.papers.items(), total=len(target_node.papers)):
        phrase_diff = cosine_similarity_embeddings([sentence_model.encode(p.title)], phrase_class_embs).reshape((-1))
        sent_diff = cosine_similarity_embeddings([sentence_model.encode(p.title)], sent_class_embs).reshape((-1))
        joint_diff = cosine_similarity_embeddings([sentence_model.encode(p.title)], joint_class_embs).reshape((-1))

        phrase_winner = phrase_diff.argmax()
        sent_winner = sent_diff.argmax()
        joint_winner = joint_diff.argmax()

        winner_idx = stats.mode([phrase_winner, sent_winner, joint_winner], keepdims=False)[0]
        winner = class_ids[winner_idx]

        class_map[winner].append(p)
        target_node.children[winner_idx].papers[p_id] = p
        
        labels.append(winner)
        scores.append(winner in p.gold)

    print(sum(scores)/len(scores))
    # we start with first level (titles and abstract)

    # if not os.path.exists(f"SeeTopic/{args.dataset}"):
    #     os.makedirs(f"SeeTopic/{args.dataset}")

    # with open(f"SeeTopic/{args.dataset}/{args.dataset}.txt", "w") as f:
    #     for p in taxo.external_collection:
    #         f.write(f"paper_title : {p.title} ; paper_abstract : {p.abstract}\n")
    
    # children_with_terms = taxo.root.getChildren(terms=True)

    # with open(f"SeeTopic/{args.dataset}/keywords_0.txt", "w") as f:
    #     for idx, c in enumerate(children_with_terms):
    #         str_c = ",".join(c[1])
    #         f.write(f"{idx}:{c[0]},{str_c}\n")
    
    # os.chdir("./SeeTopic")
    # subprocess.check_call(['./seetopic.sh', args.dataset, str(args.iters), args.model])
    # os.chdir("../")



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
    args.internal = f"{args.dataset}.txt"
    args.external = f"{args.dataset}_external.txt"
    args.groundtruth = f"groundtruth.txt"


    main(args)