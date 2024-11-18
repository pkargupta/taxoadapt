import re
import math
import os
import numpy as np
from tqdm import tqdm
import itertools
import torch
from sklearn.metrics import f1_score
from collections import deque
from sklearn.preprocessing import MultiLabelBinarizer


def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    pattern = r'^```\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', cleaned_string, flags=re.DOTALL)
    return cleaned_string.strip()


# ENRICHMENT HELPER FUNCTIONS

def rankPhrases(text, embs, class_reprs):
    ranks = rank_by_discriminative_significance(embs, class_reprs)
    ranked_tok = {text[idx]:rank for idx, rank in ranks.items()}
    return ranked_tok

def updateEnrichment(node, phrases, sentences, description, enrich_type=0):
    if node.description is None:
        node.description = description

    if enrich_type == 0: # common-sense
        for phrase in phrases:
            if phrase not in node.common_sense['phrases']:
                node.common_sense['phrases'].append(phrase)
        for sent in sentences:
            if sent not in node.common_sense['sentences']:
                node.common_sense['sentences'].append(sent)

    elif enrich_type == 1: # external corpus
        for phrase in phrases:
            if phrase not in node.external['phrases']:
                node.external['phrases'].append(phrase)
        for sent in sentences:
            if sent not in node.external['sentences']:
                node.external['sentences'].append(sent)

    else: # user corpus
        for phrase in phrases:
            if phrase not in node.corpus['phrases']:
                node.corpus['phrases'].append(phrase)
        for sent in sentences:
            if sent not in node.corpus['sentences']:
                node.corpus['sentences'].append(sent)


def expandExternal(taxo, text, embs, thresh=0, min_freq=3, percentile=99.9, classify=True, granularity='phrases'):
    if classify:
        paper_preds = {doc_id:set(['0', '1']) for doc_id in np.arange(len(taxo.collection))}

    node_text_ranks = []

    for node_id in tqdm(np.arange(0, len(taxo.label2id))):
        focus_node = taxo.root.findChild(str(node_id))

        focus_text = focus_node.getAllTerms(granularity=granularity, children=False)
        focus_text_embs = np.array([taxo.vocab[granularity][w] for w in focus_text])

        text_sim = cosine_similarity_embeddings(embs, focus_text_embs)
        avg_text_sim = average_with_harmonic_series(text_sim, axis=1)  # phrase_sim.mean(axis=1)
        percentile_sim = np.percentile(avg_text_sim, percentile)

        text_ranks = {}
        for rank, idx in enumerate(avg_text_sim.argsort()[::-1]):
            if (taxo.vocab_count[text[idx]] >= min_freq) and (avg_text_sim[idx] >= percentile_sim):
                text_ranks[rank] = (text[idx], avg_text_sim[idx])
                focus_text.append(text[idx])
                # focus_node.external[granularity].append(text[idx])
        
        node_text_ranks.append(text_ranks)
        
        if classify:
            external_focus_ranks = {doc_id:sum([1 for p in set(focus_text) if p in doc.vocabulary]) for doc_id, doc in enumerate(taxo.collection)}
            for doc_id, doc_count in external_focus_ranks.items():
                if doc_count > thresh:
                    paper_preds[doc_id].add(str(node_id))

    if classify:
        gt = [p.gold for p in taxo.collection]
        preds = list(paper_preds.values())
        print(example_f1(gt, preds))
    
    return node_text_ranks


def expandDiscriminative(taxo, text, embs, thresh=0, min_freq=3, percentile=99.9, classify=True, granularity='phrases', internal=False):
    if classify:
        paper_preds = {doc_id:set(['0','1']) for doc_id in np.arange(len(taxo.collection))}

    node_text_ranks = []

    terms_to_add = {node_id:[] for node_id in np.arange(0, len(taxo.label2id))}

    for node_id in tqdm(np.arange(0, len(taxo.label2id))):
        focus_node = taxo.root.findChild(str(node_id))
        sibling_nodes = taxo.get_sib(focus_node.node_id, granularity='emb')

        focus_text = focus_node.getAllTerms(granularity=granularity, children=False)
        focus_text_embs = np.array([taxo.vocab[granularity][w] for w in focus_text])

        text_sim = cosine_similarity_embeddings(embs, focus_text_embs)
        avg_text_sim = average_with_harmonic_series(text_sim, axis=1)  # phrase_sim.mean(axis=1)
        percentile_sim = np.percentile(avg_text_sim, percentile)

        # compute similarity to other siblings

        sibling_text = [sib.getAllTerms(granularity=granularity, children=False) for sib in sibling_nodes]
        sib_text_embs = [np.array([taxo.vocab[granularity][p] for p in phrases]) for phrases in sibling_text]

        sib_sims = [cosine_similarity_embeddings(embs, text_emb) for text_emb in sib_text_embs]
        if len(sibling_nodes):
            avg_sib_sim = np.stack([average_with_harmonic_series(sib_sim, axis=1) for sib_sim in sib_sims], axis=-1).max(axis=1)
        else:
            avg_sib_sim = np.zeros_like(avg_text_sim)

        text_ranks = {}
        for rank, idx in enumerate((avg_text_sim - avg_sib_sim).argsort()[::-1]):
            if (taxo.vocab_count[text[idx]] >= min_freq if granularity == 'phrases' else True) and (avg_text_sim[idx] >= percentile_sim) and (avg_text_sim[idx] > avg_sib_sim[idx]):
                text_ranks[rank] = (text[idx], avg_text_sim[idx])
                focus_text.append(text[idx])
                terms_to_add[node_id].append(text[idx])
        
        node_text_ranks.append(text_ranks)

        if classify:
            external_focus_ranks = {doc_id:sum([1 for p in set(focus_text) if p in doc.vocabulary]) for doc_id, doc in enumerate(taxo.collection)}
            for doc_id, doc_count in external_focus_ranks.items():
                if doc_count > thresh:
                    paper_preds[doc_id].add(str(node_id))
    
    if internal != -1:
        for node_id in tqdm(np.arange(0, len(taxo.label2id))):
            focus_node = taxo.root.findChild(str(node_id))
            if internal:
                focus_node.internal[granularity].extend(terms_to_add[node_id])
            else:
                focus_node.external[granularity].extend(terms_to_add[node_id])

    if classify:
        gt = [p.gold for p in taxo.collection]
        preds = list(paper_preds.values())
        print(example_f1(gt, preds))
    
    return node_text_ranks, gt, preds

def expandInternal(taxo, text, embs, term_to_idx, bm_score, thresh=3, min_freq=3, percentile=99.9, classify=True, granularity='phrases', reset=False):
    if classify:
        paper_preds = {doc_id:set(['0','1']) for doc_id in np.arange(len(taxo.collection))}
    
    node_text_ranks = []
    terms_to_add = {node_id:[] for node_id in np.arange(0, len(taxo.label2id))}

    for node_id in tqdm(np.arange(0, len(taxo.label2id))):
        # gather node and its siblings
        focus_node = taxo.root.findChild(str(node_id))
        sibling_nodes = taxo.get_sib(focus_node.node_id, granularity='emb')

        if reset:
            focus_node.internal[granularity] = []
            for sib in sibling_nodes:
                sib.internal[granularity] = []

        # get phrases of node and its siblings
        focus_text = focus_node.getAllTerms(granularity=granularity, children=False)
        focus_text_embs = np.array([taxo.vocab[granularity][w] for w in focus_text])
        sibling_text = [sib.getAllTerms(granularity=granularity, children=False) for sib in sibling_nodes]
        sib_text_embs = [np.array([taxo.vocab[granularity][p] for p in t]) for t in sibling_text]

        # compute target semantic similarity
        focus_sim = cosine_similarity_embeddings(embs, focus_text_embs)
        avg_focus_sim = average_with_harmonic_series(focus_sim, axis=1)  # P x 1

        # compute sibling semantic dissimilarity
        sib_sims = [cosine_similarity_embeddings(embs, s_emb) for s_emb in sib_text_embs]
        if len(sibling_nodes):
            avg_sib_sim = np.stack([average_with_harmonic_series(sib_sim, axis=1) for sib_sim in sib_sims], axis=-1).max(axis=1)
        else:
            avg_sib_sim = np.zeros_like(avg_focus_sim)

        # compute semantic rank
        target_sim_rank = {idx:rank for rank, idx in enumerate((avg_focus_sim-avg_sib_sim).argsort()[::-1])}

        # compute target co-occurrence
        target_co_ocurrence = average_with_harmonic_series(getBM25(text, focus_text, term_to_idx, bm_score), axis=1) # P x 1
        
        # compute sibling co-occurrence
        if len(sibling_nodes):
            sib_co_occurrence = np.stack([average_with_harmonic_series(getBM25(text, sib_terms, term_to_idx, bm_score), axis=1)
                                    for sib_terms in sibling_text], axis=-1).max(axis=1) # all terms x focus phrases
        else:
            sib_co_occurrence = np.zeros_like(target_co_ocurrence)
        
        # compute co-occurrence rank
        target_co_rank = {idx:rank for rank, idx in enumerate((target_co_ocurrence-sib_co_occurrence).argsort()[::-1])}

        joint_rank = compute_joint_ranking([target_sim_rank, target_co_rank]) # idx: rank
        sorted_ranks = sorted(joint_rank.items(), key=lambda x: x[1])

        final_ranks = {}
        for idx, rank in sorted_ranks:
            if rank > (1-0.01*percentile)*len(text):
                break
            if (taxo.vocab_count[text[idx]] >= min_freq) and (avg_focus_sim[idx] > avg_sib_sim[idx]) and (target_co_ocurrence[idx] > sib_co_occurrence[idx]):
                final_ranks[rank] = (text[idx], avg_focus_sim[idx], target_co_ocurrence[idx])
                focus_text.append(text[idx])
                terms_to_add[node_id].append(text[idx])
        
        node_text_ranks.append(final_ranks)


        external_focus_ranks = {doc_id:sum([1 for p in set(focus_text) if p in doc.vocabulary]) for doc_id, doc in enumerate(taxo.collection)}
        for doc_id, doc_count in external_focus_ranks.items():
            if doc_count > thresh:
                paper_preds[doc_id].add(str(node_id))
    
    for node_id in np.arange(0, len(taxo.label2id)):
        focus_node = taxo.root.findChild(str(node_id))
        focus_node.internal[granularity].extend(terms_to_add[node_id])


    gt = [p.gold for p in taxo.collection]
    preds = list(paper_preds.values())
    print(example_f1(gt, preds))

    return node_text_ranks, gt, preds

def expandSentences(taxo, term_to_idx, bm_score):
    sentence_pool = []
    sentence_phrase_pool = []
    for paper in taxo.collection:
        for s_sent, p_sent in zip(paper.sent_tokenize, paper.phrase_tokenize):
            if s_sent not in sentence_pool:
                sentence_pool.append(s_sent)
                sentence_phrase_pool.append(p_sent)

    phrase_pool_emb = [np.stack([taxo.vocab['phrases'][w] for w in s], axis=0) for s in sentence_phrase_pool] # S x P x 768
    sentence_pool_emb = np.array([taxo.vocab['sentences'][s] for s in sentence_pool])

    taxo.root.internal['sentences'] = sentence_pool
    taxo.root.internal['sent_ids'] = np.arange(len(sentence_pool))
    
    queue = deque([taxo.root])

    all_sent_ranks = {}

    while queue:
        curr_node = queue.popleft()
        # for each child, compute phrase emb and sib emb
        for child in curr_node.children:
            child.internal['sentences'] = []
            child.internal['sent_ids'] = []

            child.emb['phrase'] = np.stack([taxo.vocab['phrases'][w] 
                                            for w in child.getAllTerms(granularity='phrases', children=False)], axis=0)
            child.emb['sentence'] = np.stack([taxo.vocab['sentences'][w] 
                                            for w in child.getAllTerms(granularity='sentences', children=False)], axis=0)
        
        candidate_phrases = sentence_phrase_pool
        candidate_phrase_embs = phrase_pool_emb
        candidate_sent_embs = sentence_pool_emb

        sent_ranks = {sent_id:[] for sent_id in np.arange(len(sentence_pool_emb))} # for each candidate: list of ranks across all child nodes

        for focus_node in tqdm(curr_node.children):
            sibs = [n for n in curr_node.children if n != focus_node]

            focus_phrases = focus_node.getAllTerms(granularity='phrases', children=False)
            sibling_phrases = [sib.getAllTerms(granularity='phrases', children=False) for sib in sibs]
            
            # compute target phrase/sentence semantic similarity
            focus_phrase_sim = np.stack([cosine_similarity_embeddings(p_embs, focus_node.emb['phrase']).max(axis=0) 
                                        for p_embs in candidate_phrase_embs], axis=0) # S x [P x N] -> S x N
            avg_focus_phrase_sim = average_with_harmonic_series(focus_phrase_sim, axis=1)  # S x 1

            focus_sent_sim = cosine_similarity_embeddings(candidate_sent_embs, focus_node.emb['sentence']) # S x N
            avg_focus_sent_sim = average_with_harmonic_series(focus_sent_sim, axis=1)  # S x 1

            # compute co_occurrence with focus node
            target_co_occurrence = np.array([average_with_harmonic_series(getBM25(sent, focus_phrases, term_to_idx, bm_score).mean(axis=0)) 
                                            for sent in candidate_phrases]) # S x 1
            
            # compute sibling sentence semantic dissimilarity
            sib_phrase_sims = [np.stack([cosine_similarity_embeddings(p_embs, sib.emb['phrase']).max(axis=0)
                                        for p_embs in candidate_phrase_embs], axis=0)
                                        for sib in sibs] # siblings x sentences x P x N -> sib x sentences x N
            sib_sent_sims = [cosine_similarity_embeddings(candidate_sent_embs, sib.emb['sentence']) for sib in sibs] # siblings x sentences x sib_sents
            
            if len(sibs):
                avg_sib_phrase_sim = np.stack([average_with_harmonic_series(sib_sim, axis=1) for sib_sim in sib_phrase_sims], axis=-1).max(axis=1) # sentences x 1
                avg_sib_sent_sim = np.stack([average_with_harmonic_series(sib_sim, axis=1) for sib_sim in sib_sent_sims], axis=-1).max(axis=1) # sentences x 1
                # compute sibling co-occurrence
                sib_co_occurrence = np.array([max([average_with_harmonic_series(getBM25(sent_phrases, sib_terms, term_to_idx, bm_score).mean(axis=0)) 
                                                for sib_terms in sibling_phrases]) for sent_phrases in candidate_phrases]) # S x 1
            else:
                avg_sib_phrase_sim = np.zeros_like(avg_focus_phrase_sim)
                avg_sib_sent_sim = np.zeros_like(avg_focus_sent_sim)
                sib_co_occurrence = np.zeros_like(target_co_occurrence)
                
            # compute semantic rank
            target_sim_phrase_rank = {idx:rank for rank, idx in enumerate((avg_focus_phrase_sim-avg_sib_phrase_sim).argsort()[::-1])}
            target_sim_sent_rank = {idx:rank for rank, idx in enumerate((avg_focus_sent_sim-avg_sib_sent_sim).argsort()[::-1])}
            
            # compute co-occurrence rank
            target_co_rank = {idx:rank for rank, idx in enumerate((target_co_occurrence-sib_co_occurrence).argsort()[::-1])}

            joint_rank = compute_joint_ranking([target_sim_phrase_rank, target_sim_sent_rank, target_co_rank]) # arr idx: rank

            for idx in np.arange(len(sentence_pool)):
                sent_ranks[idx].append(joint_rank[idx])

        # filter sentences based on rank
        for node_id, focus_node in enumerate(curr_node.children):
            in_domain_phrases = focus_node.getAllTerms(granularity='phrases', children=False)

            sorted_ranks = sorted([s_id 
                                for s_id in np.arange(len(sentence_pool)) 
                                if (sent_ranks[s_id][node_id] <= min(sent_ranks[s_id])) 
                                and (len(sentence_phrase_pool[s_id]) > 5) 
                                and ((sent_ranks[s_id][node_id]) < len(sent_ranks)//2)], 
                                key=lambda x: sent_ranks[x][node_id])

            focus_node.internal['sentences'] = [sentence_pool[s_id] for s_id in sorted_ranks]
            focus_node.internal['sent_ids'] = sorted_ranks
            queue.append(focus_node)
        
        all_sent_ranks[curr_node.node_id] = sent_ranks
    
    return sentence_pool, all_sent_ranks
        



def constructTermDocMatrix(taxo, corpus):
    term_to_idx = {term:idx for idx, term in enumerate(taxo.vocab_count)}
    td_matrix = np.zeros((len(taxo.vocab_count), len(corpus))) # T x P
    co_matrix = np.zeros((len(taxo.vocab_count), len(taxo.vocab_count))) # T x T

    for p_id, paper in tqdm(enumerate(corpus), total=len(corpus)):
        term_ids = []
        term_freqs = []
        for term in paper.vocabulary:
            term_ids.append(term_to_idx[term])
            term_freqs.append(paper.vocabulary[term])

        xy = np.array(np.meshgrid(term_ids, term_ids)).T.reshape((-1,2))

        td_matrix[term_ids, p_id] = term_freqs
        co_matrix[xy[:, 0], xy[:, 1]] += 1
    
    return term_to_idx, td_matrix, co_matrix

def computeBM25Cog(co_matrix, co_avg, k=1.2, b=2):
    co_score = co_matrix * (k + 1) / (co_matrix + k * (1 - b + b * (co_matrix.sum(axis=1, keepdims=True) / co_avg)))
    query_sum = co_matrix.astype(bool).sum(axis=0, keepdims=True)
    df_factor = np.divide(np.log2(1 + len(co_matrix) - query_sum), np.log2(1 + query_sum))
    bm_score = co_score * df_factor
    return bm_score

def computeBM25CogTemp(co_matrix, co_avg, k=1.2, b=2):
    co_score = co_matrix * (k + 1) / (co_matrix + k * (1 - b + b * (co_matrix.sum(axis=0, keepdims=True) / co_avg)))
    query_sum = co_matrix.astype(bool).sum(axis=1, keepdims=True)
    df_factor = np.divide(np.log2(1 + len(co_matrix) - query_sum), np.log2(1 + query_sum))
    bm_score = co_score * df_factor
    return bm_score

def getBM25(term, query, term_to_idx, bm_score):

    if type(term) == list:
        t_id = [term_to_idx[t] for t in term if t in term_to_idx]
        q_id = [term_to_idx[q] for q in query if q in term_to_idx]

        return bm_score[np.ix_(t_id, q_id)]
    else:
        if (term not in term_to_idx) or (query not in term_to_idx):
            return 0

        t_id = term_to_idx[term]
        q_id = term_to_idx[query]

        return bm_score[t_id, q_id]

# def getBM25(term, query, term_to_idx, co_matrix, co_avg, k=1.2, b=1):
    
#     if type(term) != list:
#         term = [term]
#     if type(query) != list:
#         query = [query]

#     t_id = [term_to_idx[t] for t in term if t in term_to_idx]
#     q_id = [term_to_idx[q] for q in query if q in term_to_idx]

#     if (len(t_id) == 0) or (len(q_id) == 0):
#         return 0

#     tq_co_occur = co_matrix[np.ix_(t_id, q_id)]

#     term_co = co_matrix[t_id].sum(axis=1, keepdims=True)
#     query_co = co_matrix[q_id].astype(bool).sum(axis=1)

#     co_score = tq_co_occur * (k + 1) / (tq_co_occur + k * (1 - b + b * (term_co / co_avg)))
#     df_factor = np.log2(1 + len(co_matrix) - query_co) / np.log2(1 + query_co)
#     bm_score = co_score * df_factor

#     return bm_score


# def compareClasses(w, taxo, node, granularity='phrases'):
#     if type(w) == str:
#         embs = np.array([taxo.vocab[granularity][w]])
#     else:
#         embs = np.array([taxo.vocab[granularity][item] for item in w])
#     sibs = taxo.get_sib(node.node_id, granularity)
#     if granularity == 'phrases':
#         curr_node_sim = cosine_similarity_embeddings(embs, 
#                                                      sentence_model.encode([node.label] 
#                                                                            + sibs))
#     else:
#         curr_node_sim = cosine_similarity_embeddings(embs, 
#                                                      sentence_model.encode([node.label + " : " + node.description] 
#                                                                            + sibs))
    
#     if len(sibs) == 0:
#         decision = np.array([True] * len(curr_node_sim)) #curr_node_sim[:, 0] >= curr_node_sim[:, 0].mean() # get top 50% of items if there are no other siblings
#         return embs, curr_node_sim[:, 0], decision
#     else:
#         sim_diff = curr_node_sim[:, 0] - curr_node_sim[:, 1:].max(axis=1)
#         decision = curr_node_sim[:, 0] > curr_node_sim[:, 1:].max(axis=1)
#         decision[0] = True
#         return embs, sim_diff, decision
    
def compareClassesEmbs(w, taxo, node, granularity='phrases', parent_weight=0.0):
    if type(w) == str:
        embs = np.array([taxo.vocab[granularity][w]])
    else:
        embs = np.array([taxo.vocab[granularity][item] for item in w])
    
    sibs = [sib for sib in taxo.get_sib(node.node_id, 'emb')]
    compute_with_parent = lambda focus_node: np.average([focus_node.parents[0].emb[granularity], focus_node.emb[granularity]], 
                                                        weights=[parent_weight, 1.0-parent_weight], axis=0) if len(focus_node.parents[0].emb) > 0 else focus_node.emb[granularity]
    class_embs = [compute_with_parent(n) for n in [node] + sibs]
    curr_node_sim = cosine_similarity_embeddings(embs, class_embs)
    
    if len(sibs) == 0:
        decision = np.array([True] * len(curr_node_sim)) # curr_node_sim[:, 0] >= curr_node_sim[:, 0].mean() # get top 50% of items if there are no other siblings
        return embs, curr_node_sim[:, 0], decision
    else:
        sim_diff = curr_node_sim[:, 0] - curr_node_sim[:, 1:].max(axis=1)
        decision = curr_node_sim[:, 0] > curr_node_sim[:, 1:].max(axis=1)
        return embs, sim_diff, decision

# ranking helper functions

def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


def filter_by_class_discriminative_significance(embeddings, class_embeddings, class_id):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    class_similarities = similarities[:, class_id]
    other_dissimilarity = np.concatenate([similarities[:, :class_id], similarities[:, class_id+1:]], axis=1)
    significance_score = class_similarities - other_dissimilarity.max(axis=1)
    filtered_scores = [i for i in np.argsort(-np.array(significance_score)) if significance_score[i] > 0]

    # significance_score = [np.max(np.sort(similarity)[-2:]) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(filtered_scores)}
    return significance_ranking

def rank_by_class_discriminative_significance(embeddings, class_embeddings, class_id):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    if similarities.shape[1] > 1:
        class_similarities = similarities[:, class_id]
        other_dissimilarity = np.concatenate([similarities[:, :class_id], similarities[:, class_id+1:]], axis=1)
        significance_score = class_similarities - other_dissimilarity.max(axis=1)
    else:
        significance_score = similarities[:, class_id]
    # significance_score = [np.max(np.sort(similarity)[-2:]) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking

def rank_by_max_discriminative_significance(embeddings, class_embeddings):
    similarities = np.stack([cosine_similarity_embeddings(embeddings, embs).max(axis=1) for embs in class_embeddings], axis=1)
    significance_score = np.ptp(np.sort(similarities, axis=1)[:, -2:], axis=1)
    # significance_score = [np.max(np.sort(similarity)[-2:]) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking

def rank_by_discriminative_significance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = np.ptp(np.sort(similarities, axis=1)[:, -2:], axis=1)
    # significance_score = [np.max(np.sort(similarity)[-2:]) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking

def rank_by_significance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(similarity) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking

def rank_by_insignificance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(similarity) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(np.array(significance_score)))}
    return significance_ranking

def rank_by_lexical(phrases, mapped, unmapped):
    idf = lambda w: np.log((1 + len(mapped))/(1 + np.sum([1 for paper in mapped if w in paper.vocabulary])))
    w_idf = {term:idf(term) for term in phrases}
    tf = {term:len([term in p.vocabulary for p in unmapped]) for term in phrases}

    lexical_score = [tf[term]*w_idf[term] for term in phrases]
    lexical_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(lexical_score)))}

    return lexical_ranking


def rank_by_relation(embeddings, class_embeddings):
    relation_score = cosine_similarity_embeddings(embeddings, [np.average(class_embeddings, axis=0)]).reshape((-1))
    relation_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(relation_score)))}
    return relation_ranking


def mul(l):
    m = 1
    for x in l:
        m *= x + 1
    return m


def average_with_harmonic_series(representations, axis=0):
    if type(representations) == list:
        representations = np.array(representations)
    dim = representations.shape[axis]
    weights = [0.0] * dim
    for i in range(dim):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=axis)

def compute_joint_ranking(rankings):
    if len(rankings) == 0:
        assert False
    if type(rankings[0]) == type(0):
        rankings = [rankings]
    
    rankings_num = len(rankings)
    rankings_len = len(rankings[0])
    assert all(len(rankings[i]) == rankings_len for i in range(rankings_num))
    total_score = []
    for i in range(rankings_len):
        total_score.append(mul(ranking[i] for ranking in rankings))

    total_ranking = {i: r for r, i in enumerate(np.argsort(np.array(total_score)))}

    return total_ranking

def weights_from_ranking(rankings):
    if len(rankings) == 0:
        assert False
    if type(rankings[0]) == type(0):
        rankings = [rankings]
    rankings_num = len(rankings)
    rankings_len = len(rankings[0])
    assert all(len(rankings[i]) == rankings_len for i in range(rankings_num))
    total_score = []
    for i in range(rankings_len):
        total_score.append(mul(ranking[i] for ranking in rankings))

    total_ranking = {i: r for r, i in enumerate(np.argsort(np.array(total_score)))}

    # print("TOTAL RANKING:", total_ranking)
    # print("OG RANKING:", rankings[0])
    # NEW: WE WANT TO COMMENT THIS OUT BECAUSE CERTAIN WORDS MIGHT BE REPEATED AND THUS HAVE THE SAME RANK
    # if rankings_num == 1:
    #     assert all(total_ranking[i] == rankings[0][i] for i in total_ranking.keys())
    weights = [0.0] * rankings_len
    for i in range(rankings_len):
        weights[i] = 1. / (total_ranking[i] + 1)
    return weights


def weight_sentence_with_attention(vocab, tokenized_text, contextualized_word_representations, class_representations,
                                   attention_mechanism):
    assert len(tokenized_text) == len(contextualized_word_representations)

    contextualized_representations = []
    static_representations = []

    static_word_representations = vocab["static_word_representations"]
    word_to_index = vocab["word_to_index"]
    for i, token in enumerate(tokenized_text):
        if token in word_to_index:
            static_representations.append(static_word_representations[word_to_index[token]])
            contextualized_representations.append(contextualized_word_representations[i])
    if len(contextualized_representations) == 0:
        print("Empty Sentence (or sentence with no words that have enough frequency)")
        return np.average(contextualized_word_representations, axis=0)

    significance_ranking = rank_by_significance(contextualized_representations, class_representations)
    relation_ranking = rank_by_relation(contextualized_representations, class_representations)
    significance_ranking_static = rank_by_significance(static_representations, class_representations)
    relation_ranking_static = rank_by_relation(static_representations, class_representations)
    if attention_mechanism == "none":
        weights = [1.0] * len(contextualized_representations)
    elif attention_mechanism == "significance":
        weights = weights_from_ranking(significance_ranking)
    elif attention_mechanism == "relation":
        weights = weights_from_ranking(relation_ranking)
    elif attention_mechanism == "significance_static":
        weights = weights_from_ranking(relation_ranking)
    elif attention_mechanism == "relation_static":
        weights = weights_from_ranking(relation_ranking)
    elif attention_mechanism == "mixture":
        weights = weights_from_ranking((significance_ranking,
                                        relation_ranking,
                                        significance_ranking_static,
                                        relation_ranking_static))
    else:
        assert False
    return np.average(contextualized_representations, weights=weights, axis=0)


def weight_sentence(model,
                    vocab,
                    tokenization_info,
                    class_representations,
                    attention_mechanism,
                    layer):
    
    tokenized_text, tokenized_to_id_indicies, tokenids_chunks = tokenization_info
    contextualized_word_representations = handle_sentence(model, layer, tokenized_text, tokenized_to_id_indicies,
                                                          tokenids_chunks)
    document_representation = weight_sentence_with_attention(vocab, tokenized_text, contextualized_word_representations,
                                                             class_representations, attention_mechanism)
    return document_representation


# EVALUATION HELPER FUNCTIONS

def precision_at_k(preds, gts, k=1):
    assert len(preds) == len(gts), "number of samples mismatch"
    p_k = 0.0
    for pred, gt in zip(preds, gts):
        p_k += ( len(set(pred[:k]) & set(gt)) / k ) 
    p_k /= len(preds)
    return p_k


def mrr(all_ranks):
    """ Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    """
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions)
    return (1.0 / scaled_rank_positions).mean()


def example_f1(trues, preds):
    """
    trues: a list of true classes
    preds: a list of model predicted classes
    """
    f1_list = []
    for t, p in zip(trues, preds):
        f1 = 2 * len(set(t) & set(p)) / (len(t) + len(p))
        f1_list.append(f1)
    return np.array(f1_list).mean()

def f1_scores(gt, preds):
    # Example multi-label true labels and predictions
    y_true = gt  # True labels
    y_pred = preds  # Model predictions

    # Use MultiLabelBinarizer to convert to binary format
    mlb = MultiLabelBinarizer()
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    # Calculate F1-Macro and F1-Micro scores
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro')
    f1_micro = f1_score(y_true_bin, y_pred_bin, average='micro')

    print(f'F1-Macro Score: {f1_macro}')
    print(f'F1-Micro Score: {f1_micro}')