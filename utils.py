import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from model_definitions import llama_8b_model
from prompts import phrase_filter_init_prompt, phrase_filter_prompt
import re

def filter_phrases(topics, phrases, word2emb, other_parents):
    messages = [
            {"role": "system", "content": phrase_filter_init_prompt},
            {"role": "user", "content": phrase_filter_prompt(topics, phrases, other_parents)}]
        
    model_prompt = llama_8b_model.tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_generation_prompt=True)

    terminators = [
        llama_8b_model.tokenizer.eos_token_id,
        llama_8b_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = llama_8b_model(
        model_prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id=llama_8b_model.tokenizer.eos_token_id
    )
    message = outputs[0]["generated_text"][len(model_prompt):]

    print(message)

    phrases = re.findall(r'.*_filtered:\s*\[*(.*)\]*', message, re.IGNORECASE)[0]

    phrases = re.findall(r'([\w.-]+)[,\'"]*', phrases, re.IGNORECASE)

    iv_phrases = []
    vocab = list(word2emb.keys())
    mod_vocab = [w.replace("-", " ").replace("_", " ") for w in vocab]

    for p in phrases:
        if p not in word2emb.keys():
            if p.replace("-", " ").replace("_", " ") in mod_vocab:
                iv_phrases.append(vocab[mod_vocab.index(p.replace("-", " ").replace("_", " "))])
            else:
                print(p, "not found!")
        else:
            iv_phrases.append(p)

    return iv_phrases

# def filter_phrases_NEW(topics, phrases, other_parents):
#     messages = [
#             {"role": "system", "content": phrase_filter_init_prompt},
#             {"role": "user", "content": phrase_filter_prompt(topics, f"{topics}: {phrases}\n", other_parents)}]
        
#     model_prompt = llama_8b_model.tokenizer.apply_chat_template(messages, 
#                                                     tokenize=False, 
#                                                     add_generation_prompt=True)

#     terminators = [
#         llama_8b_model.tokenizer.eos_token_id,
#         llama_8b_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = llama_8b_model(
#         model_prompt,
#         max_new_tokens=1024,
#         eos_token_id=terminators,
#         do_sample=False,
#         pad_token_id=llama_8b_model.tokenizer.eos_token_id
#     )
#     message = outputs[0]["generated_text"][len(model_prompt):]

#     print(message)

#     invalid_phrases = re.findall(r'.*_invalid_subtopics:\s*\[*(.*)\]*', message, re.IGNORECASE)[0]

#     invalid_phrases = re.findall(r'([\w.-]+)[,\'"]*', invalid_phrases, re.IGNORECASE)

#     valid_phrases = phrases.copy()
#     mod_phrases = [w.replace("-", " ").replace("_", " ") for w in valid_phrases]
    

#     for p in invalid_phrases:
#         if p in valid_phrases:
#             valid_phrases.remove(p)
#             mod_phrases.remove(p.replace("-", " ").replace("_", " "))
#         elif p in mod_phrases:
#             valid_phrases.remove(valid_phrases[mod_phrases.index(p)])
#             mod_phrases.remove(p)
    
#     return valid_phrases

## NODE-ORIENTED SENTENCE REPRESENTATIONS ##

def cosine_similarity_embeddings(emb_a, emb_b):
    return np.dot(emb_a, np.transpose(emb_b)) / np.outer(np.linalg.norm(emb_a, axis=1), np.linalg.norm(emb_b, axis=1))


def rank_by_significance(embeddings, class_embeddings):
    similarities = cosine_similarity_embeddings(embeddings, class_embeddings)
    significance_score = [np.max(similarity) for similarity in similarities]
    significance_ranking = {i: r for r, i in enumerate(np.argsort(-np.array(significance_score)))}
    return significance_ranking


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
    if rankings_num == 1:
        assert all(total_ranking[i] == rankings[0][i] for i in total_ranking.keys())
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
