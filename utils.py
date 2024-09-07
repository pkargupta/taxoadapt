import re
import os
import numpy as np
import torch
from model_definitions import sentence_model
from sklearn.metrics import f1_score

def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
    return cleaned_string.strip()


# ENRICHMENT HELPER FUNCTIONS

def updateEnrichment(node, phrases, sentences, enrich_type=0):
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

def compareClasses(w, taxo, node, granularity='phrases'):
    if type(w) == str:
        embs = np.array([taxo.vocab[granularity][w]])
    else:
        embs = np.array([taxo.vocab[granularity][item] for item in w])
    sibs = taxo.get_sib(node.node_id, granularity)
    if granularity == 'phrases':
        curr_node_sim = cosine_similarity_embeddings(embs, 
                                                     sentence_model.encode([node.label] 
                                                                           + sibs))
    else:
        curr_node_sim = cosine_similarity_embeddings(embs, 
                                                     sentence_model.encode([node.label + " : " + node.description] 
                                                                           + sibs))
    
    if len(sibs) == 0:
        decision = np.array([True] * len(curr_node_sim)) #curr_node_sim[:, 0] >= curr_node_sim[:, 0].mean() # get top 50% of items if there are no other siblings
        return embs, curr_node_sim[:, 0], decision
    else:
        sim_diff = curr_node_sim[:, 0] - curr_node_sim[:, 1:].max(axis=1)
        decision = curr_node_sim[:, 0] > curr_node_sim[:, 1:].max(axis=1)
        decision[0] = True
        return embs, sim_diff, decision
    
def compareClassesEmbs(w, taxo, node, granularity='phrases'):
    if type(w) == str:
        embs = np.array([taxo.vocab[granularity][w]])
    else:
        embs = np.array([taxo.vocab[granularity][item] for item in w])
    
    sibs = [sib.emb[granularity] for sib in taxo.get_sib(node.node_id, 'emb')]
    curr_node_sim = cosine_similarity_embeddings(embs, [node.emb[granularity]] + sibs)
    
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


def average_with_harmonic_series(representations):
    weights = [0.0] * len(representations)
    for i in range(len(representations)):
        weights[i] = 1. / (i + 1)
    return np.average(representations, weights=weights, axis=0)


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