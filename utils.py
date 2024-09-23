import re
import os
import numpy as np
from tqdm import tqdm
import itertools
import torch
from model_definitions import sentence_model
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer


def clean_json_string(json_string):
    pattern = r'^```json\s*(.*?)\s*```$'
    cleaned_string = re.sub(pattern, r'\1', json_string, flags=re.DOTALL)
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
                if internal != -1:
                    if internal:
                        focus_node.internal[granularity].append(text[idx])
                    else:
                        focus_node.external[granularity].append(text[idx])
        
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
    
    return node_text_ranks, gt, preds

def expandInternal(taxo, text, embs, term_to_idx, bm_score, thresh=3, min_freq=3, percentile=99.9, classify=True, granularity='phrases'):
    if classify:
        paper_preds = {doc_id:set(['0','1']) for doc_id in np.arange(len(taxo.collection))}
    
    node_text_ranks = []

    for node_id in tqdm(np.arange(0, len(taxo.label2id))):
        # gather node and its siblings
        focus_node = taxo.root.findChild(str(node_id))
        sibling_nodes = taxo.get_sib(focus_node.node_id, granularity='emb')

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
        target_co_ocurrence = np.array([average_with_harmonic_series([getBM25(term, focus_term, term_to_idx, bm_score) for focus_term in focus_text])
                            for term in text]) # P x 1
        
        # compute sibling co-occurrence
        if len(sibling_nodes):
            sib_co_occurrence = np.array([max([average_with_harmonic_series([getBM25(term, sib_term, term_to_idx, bm_score) for sib_term in sib_terms])
                                    for sib_terms in sibling_text])
                                    for term in text]) # all terms x focus phrases
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
                focus_node.internal[granularity].append(text[idx])
        
        node_text_ranks.append(final_ranks)


        external_focus_ranks = {doc_id:sum([1 for p in set(focus_text) if p in doc.vocabulary]) for doc_id, doc in enumerate(taxo.collection)}
        for doc_id, doc_count in external_focus_ranks.items():
            if doc_count > thresh:
                paper_preds[doc_id].add(str(node_id))


    gt = [p.gold for p in taxo.collection]
    preds = list(paper_preds.values())
    print(example_f1(gt, preds))

    return node_text_ranks, gt, preds


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
    co_score = co_matrix * (k + 1) / (co_matrix + k * (1 - b + b * (co_matrix.sum(axis=0, keepdims=True) / co_avg)))
    query_sum = co_matrix.astype(bool).sum(axis=1, keepdims=True)
    # df_factor = np.log2(np.divide((len(co_matrix) - query_sum + 0.5), (query_sum + 0.5)))
    df_factor = np.divide(np.log2(1 + len(co_matrix) - query_sum), np.log2(1 + query_sum))
    # df_factor = np.log2(0.5 + query_sum) / math.log(1 + co_matrix.shape[0], 2)
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