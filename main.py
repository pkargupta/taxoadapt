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

def commonSenseEnrich(taxo, dict_str, batch=True):
    root_node = taxo.root
    # phrase and sentence-level enrichment

    # construct prompts for each node
    prompts = []
    nodes = []
    queue = deque([root_node])
    while queue:
        current_node = queue.popleft()
        nodes.append(current_node)
        sibs = [i.label for i in current_node.parents[0].children if i != current_node]
        prompts.append(constructPrompt(init_enrich_prompt, main_simple_enrich_prompt(taxo, current_node, sibs), api=False))

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

    for idx, c in tqdm(enumerate(output_dict), total=len(output_dict)):
        c['example_key_phrases'] = [p.lower().replace(' ', '_') for p in c['example_key_phrases']]

        current_node = nodes[idx]

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
    taxo = Taxonomy(args.data_dir)

    taxo_dict = taxo.toDict(cur_node=taxo.root)
    with open(f'datasets/{args.dataset}/initial.json', 'w') as fp:
        json.dump(taxo_dict, fp, indent=4)
    dict_str = json.dumps(taxo_dict, indent=4)

    # first do common sense phrase, sentence enrichment on taxonomy nodes
    enrich_start = time.time()
    prompts, outputs, all_common_phrases, all_common_sentences = commonSenseEnrich(taxo.root, dict_str, True)
    enrich_end = time.time()

    if all_common_phrases:
        # update vocabulary/embeddings
        taxo.updateVocab(all_common_phrases, 'phrases')
        taxo.updateVocab(all_common_sentences, 'sentences')
        print(f"Time taken: {enrich_end - enrich_start} seconds ({(enrich_end - enrich_start)/60} minutes)")
    
    updated_dict = taxo.toDict(cur_node=taxo.root)
    with open(f'datasets/{args.dataset}/enriched.json', 'w') as fp:
        json.dump(updated_dict, fp, indent=4)


    print("########### PRE-PROCESSING BOTH CORPORA ###########")

    # add common-sense phrases to AutoPhrase
    with open("preprocessing/AutoPhrase/data/EN/wiki_quality_orig.txt", "r", encoding='utf-8') as f:
        all_phrases = [w.strip() for w in f.readlines() if " " in w.strip()]
        for a in list(taxo.label2id.keys()) + all_common_phrases:
            all_phrases.append(a.strip().replace("_", " "))

    with open("preprocessing/AutoPhrase/data/EN/wiki_quality.txt", "w", encoding='utf-8') as f:
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


    collection, external_collection = taxo.createCollections(args)


    print("########### EXTERNAL + INTERNAL ENRICHMENT ###########")

    print("encoding sentences...")
    external_sentences = list(set([sentence for paper in external_collection for sentence in paper.sent_tokenize]))
    internal_sentences = list(set([sentence for paper in collection for sentence in paper.sent_tokenize]))
    taxo.updateVocab(external_sentences + internal_sentences, 'sentences')
    # sent2emb = {sent:idx for idx, sent in enumerate(external_sentences)}
    # external_sent_emb = {idx:emb for idx, emb in  enumerate(sentence_model.encode(external_sentences))}

    print("encoding phrases...")
    external_phrases = list(set([phrase for paper in external_collection for sentence in paper.phrase_tokenize for phrase in sentence]))
    internal_phrases = list(set([phrase for paper in collection for sentence in paper.phrase_tokenize for phrase in sentence]))
    taxo.updateVocab(external_phrases + internal_phrases, 'phrases')

    taxo.graph.external['phrases'] = external_phrases
    taxo.graph.external['sentences'] = external_sentences
    taxo.graph.internal['phrases'] = internal_phrases
    taxo.graph.internal['sentences'] = internal_sentences

    print("external term enrichment...")
    phrase_pool = external_phrases
    pool_emb = np.array([taxo.vocab['phrases'][w] for w in phrase_pool])

    node_external_phrase_ranks, gt, preds = expandDiscriminative(taxo, phrase_pool, pool_emb, internal=False)
    f1_scores(gt, preds)

    print("internal term enrichment...")
    term_to_idx, td_matrix, co_matrix = constructTermDocMatrix(taxo, collection + external_collection)
    co_avg = np.true_divide(co_matrix.sum(),(co_matrix!=0).sum())

    phrase_pool = internal_phrases
    pool_emb = np.array([taxo.vocab['phrases'][w] for w in phrase_pool])
    bm_score = computeBM25Cog(co_matrix, co_avg, k=1.2, b=2)

    node_internal_phrase_ranks, gt, preds = expandInternal(taxo, phrase_pool, pool_emb, term_to_idx, bm_score)
    f1_scores(gt, preds)

    print("internal sentence enrichment...")
    expandSentences(taxo, term_to_idx, bm_score)

    print("classification...")
    # constructing class embeddings
    class_embs = []
    for node_id in tqdm(taxo.id2label):
        curr_node = taxo.root.findChild(node_id)
        curr_node.emb['sentence'] = average_with_harmonic_series(np.stack([taxo.vocab['sentences'][w] 
                                        for w in curr_node.getAllTerms(granularity='sentences', children=True)], axis=0), axis=0)
        class_embs.append(curr_node.emb['sentence'])
    




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