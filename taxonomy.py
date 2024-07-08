import os
import re
import numpy as np
from collections import deque, Counter
import json
from model_definitions import llama_8b_model, sentence_model
from prompts import depth_expansion_init_prompt, depth_expansion_prompt
from utils import average_with_harmonic_series, cosine_similarity_embeddings, mul
from utils import rank_by_discriminative_significance, rank_by_significance, rank_by_insignificance, rank_by_lexical, weights_from_ranking
from nltk.corpus import stopwords


class Paper:
    def __init__(self, taxo, id, title, abstract, text):
        self.taxo = taxo
        self.id = id
        self.title = title
        self.abstract = abstract
        self.text = text # f"title : {title}; abstract: {abstract}"
        self.split_text = self.text.split(" ")
        
        
        self.sentences = []
        self.tokenized = []

        self.nodes = {} # path: score
        self.node_terms = {} # key: node path; value: paper-specific terms relevant to node

        self.vocabulary = dict(Counter(self.split_text))

        self.phrase_emb = None
        self.emb = None

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.id == other.id)

    def __repr__(self) -> str:
        return self.text
    
    def updateVocab(self, terms):
        updated = False
        for t in terms:
            mod_title = re.sub(fr" {t.replace('_', '.')} ", f" {t} ", self.title)
            mod_abstract = re.sub(fr" {t.replace('_', '.')} ", f" {t} ", self.abstract)
            if (mod_title != self.title) or (mod_abstract != self.abstract):
                self.title = mod_title
                self.abstract = mod_abstract
                updated = True
                
        
        if updated:
            self.text = f"title : {self.title} ; abstract : {self.abstract}"
            self.sentences = self.text.split(" . ")
            self.tokenized = [sent.split() for sent in self.sentences]
            self.split_text = self.text.split(" ")
            self.vocabulary = dict(Counter(self.split_text))
        
        return
    
    def addNodeTerms(self, node, terms):
        self.updateVocab(terms)

        # only consider terms that are in the vocab AND (1) this node has not been added yet OR (2) are not already in the node_terms 
        filtered_terms = list(filter(lambda w: (w in self.vocabulary.keys()) 
                                     and ((node.path not in self.node_terms.keys()) 
                                          or (w not in self.node_terms[node.path])), terms))
        score = np.sum([self.vocabulary[ele] for ele in filtered_terms])
        if (node.path in self.node_terms.keys()):
            self.node_terms[node.path].extend(filtered_terms)
            self.nodes[node.path] += score

        else:
            self.node_terms[node.path] = filtered_terms
            self.nodes[node.path] = score
        
        return score
    
    def rankPhrases(self, class_reprs):
        iv_phrases = [w for w in self.vocabulary if w in self.taxo.static_emb]
        phrase_reprs = np.concatenate([self.taxo.static_emb[w].reshape((-1, 768)) for w in iv_phrases], axis=0)
        ranks = rank_by_discriminative_significance(phrase_reprs, class_reprs)
        ranked_tok = {iv_phrases[idx]:rank for idx, rank in ranks.items()}
        return ranked_tok
    
    def rankSentences(self, class_reprs):
        ranked_phrases = self.rankPhrases(class_reprs)

        sent_avg_weights = []
        sent_reprs = []
        for sent in self.tokenized:
            phrase_reprs = []
            phrase_ranks = {}
            p_id = 0
            for phrase in sent:
                if (phrase in self.taxo.static_emb) and (phrase not in stopwords.words('english')):
                    phrase_reprs.append(self.taxo.static_emb[phrase])
                    if phrase not in self.vocabulary:
                        print(self.id)
                    phrase_ranks[p_id] = ranked_phrases[phrase]
                    p_id += 1
            
            if len(phrase_ranks) > 0:
                phrase_weights = weights_from_ranking({k: v for k, v in sorted(phrase_ranks.items(), key=lambda item: item[1])})
                sent_avg_weights.append(np.mean(list(phrase_ranks.values())))

                # sent_reprs.append(sentence_model.encode(" ".join(sent)))
                sent_reprs.append(np.average(phrase_reprs, weights=phrase_weights, axis=0))
        
        sent_avg_ranks = {idx:rank for rank, idx in enumerate(np.argsort(sent_avg_weights))}
        sent_dis_ranks = rank_by_discriminative_significance(sent_reprs, class_reprs)
        return sent_reprs, sent_avg_ranks, sent_dis_ranks
    
    def computePaperEmb(self, class_reprs=None, phrase=True):
        if phrase:
            # sent-based repr
            sent_reprs, sent_avg_ranks, sent_dis_ranks = self.rankSentences(class_reprs)
            weights = weights_from_ranking([sent_avg_ranks, sent_dis_ranks])
            self.phrase_emb = np.average(sent_reprs, weights=weights, axis=0)
            return self.phrase_emb
        else:
            # SPECTER-based
            self.emb = sentence_model.encode(self.title + "[SEP]" + self.abstract)
            return self.emb


        # term-based repr
        # ranked_phrases = self.rankPhrases(class_phrase_reprs)

        # phrase_ranks = {}
        # phrase_reprs = []
        # p_id = 0
        # for sent in self.sentences:
        #     for phrase in sent:
        #         if phrase in self.taxo.word2emb:
        #             phrase_reprs.append(self.taxo.word2emb[phrase])
        #             phrase_ranks[p_id] = ranked_phrases[phrase]
        #             p_id += 1
        
        # if len(phrase_ranks) > 0:
        #     phrase_weights = weights_from_ranking({k: v for k, v in sorted(phrase_ranks.items(), 
        #                                                             key=lambda item: item[1])})
        #     self.emb =np.average(phrase_reprs, weights=phrase_weights, axis=0)



        return self.emb


class Node:
    def __init__(self, taxo, label, seeds=[], description=None, parent=None):
        self.taxo = taxo
        self.label = label
        self.model = llama_8b_model

        self.parent = parent

        if parent is None:
            self.path = self.label
        else:
            self.path = parent.getPath() + "->" + self.label

        self.children = []

        self.seeds = seeds
        self.desc = description

        self.papers = [] # paper
        self.paper_scores = {} # paper_id: score
        self.density = 0

        self.mined_terms = []
        self.all_node_terms = [s.lower().replace(" ", "_") for s in [self.label] + seeds]

        # get initial pool of papers
        for paper in self.taxo.collection:
            freq = paper.addNodeTerms(self, self.all_node_terms)
            if freq >= self.taxo.min_freq:
                if paper.id not in self.papers:
                    self.papers.append(paper)
                self.paper_scores[paper.id] = freq

        self.papers = sorted(self.papers, key=lambda x: self.paper_scores[x.id], reverse=True)
        
        self.all_paper_terms = set()
        self.phrase_emb = None
        self.emb = None

    def __repr__(self) -> str:
        return self.label
    
    def getPath(self):
        return self.path

    def getChildren(self, terms=False):
        if terms:
            return [(c.label, c.all_node_terms) for c in self.children]
        else:
            return [c.label for c in self.children]
    
    def updateNodeEmb(self, phrase=False):
        # self_repr = average_with_harmonic_series([self.taxo.word2emb[w] for w in [self.label] + self.seeds + self.mined_terms])
        # child_repr = [c.emb for c in self.children if c.emb is not None]
        # child_repr = np.mean(child_repr, axis=0) if len(child_repr) > 0 else []
        # self.emb = average_with_harmonic_series([self_repr] + child_repr)

        # PHRASE-BASED
        if phrase:
            self.phrase_emb = average_with_harmonic_series(np.concatenate([self.taxo.static_emb[w].reshape((1,-1)) 
                                                        for w in self.all_node_terms 
                                                        if w in self.taxo.static_emb], axis=0))
            return self.phrase_emb
        else:
            # SPECTER-BASED
            self.emb = average_with_harmonic_series([sentence_model.encode(p.title + "[SEP]" + p.abstract) 
                                                    for p in self.papers])

            return self.emb
    
    def addChild(self, label, seeds, desc):
        child_node = Node(self.taxo, label, seeds, desc, parent=self)
        self.children.append(child_node)
        return child_node
    
    def addChildren(self, labels, seeds, descriptions):
        for l, s, d in zip(labels, seeds, descriptions):
            child_node = Node(self.taxo, l, s, d, parent=self)
            self.children.append(child_node)
            self.addTerms(s, mined=False, addToParent=True)

        return self.children
    
    def addTerms(self, terms, mined=False, addToParent=False):
        new_terms = []
        for t in terms:
            mod_t = t.lower().replace(" ", "_")
            if mod_t not in self.all_node_terms:
                self.all_node_terms.append(mod_t)
                if mined:
                    self.mined_terms.append(mod_t)
                new_terms.append(mod_t)

        for paper in self.taxo.collection:
            freq = paper.addNodeTerms(self, new_terms)
            if freq >= self.taxo.min_freq: # min frequency
                if paper not in self.papers:
                    self.papers.append(paper)
                self.paper_scores[paper.id] = paper.nodes[self.path]

        self.papers = sorted(self.papers, key=lambda x: self.paper_scores[x.id], reverse=True)

        # if (self.taxo.word2emb is not None) and (self.parent is not None):
        #     self.updateNodeEmb()

        if addToParent and (self.parent is not None):
            self.parent.addTerms(terms, addToParent=addToParent)

    
    def genCommonSenseChildren(self, global_taxo=None, k=5, num_terms=10):
        messages = [
            {"role": "system", "content": depth_expansion_init_prompt(global_taxo, k, num_terms)},
            {"role": "user", "content": depth_expansion_prompt(self.label, num_terms)}]
        
        model_prompt = self.model.tokenizer.apply_chat_template(messages, 
                                                        tokenize=False, 
                                                        add_generation_prompt=True)

        terminators = [
            self.model.tokenizer.eos_token_id,
            self.model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model(
            model_prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=False,
            pad_token_id=self.model.tokenizer.eos_token_id
        )
        message = outputs[0]["generated_text"][len(model_prompt):]

        # parse for children
        labels = [i.lower().replace(" ", "_") for i in re.findall(r'label:\s*(.*)', message, re.IGNORECASE)]
        descriptions = re.findall(r'description:\s*(.*)', message, re.IGNORECASE)
        seeds = [i.split(", ") for i in re.findall(r'terms:\s*\[(.*)\]', message, re.IGNORECASE)]

        return self.addChildren(labels, seeds, descriptions)
    
    def addPaper(self, paper):
        if paper not in self.papers:
            self.papers.append(paper)
            self.paper_scores[paper.id] = paper.nodes[self.path]
        if (self.path in paper.nodes) and (paper.nodes[self.path] > 0):
            self.all_paper_terms.update(paper.node_terms[self.path])
        # update density (TODO: open problem):
        self.density = len(self.papers)/len(self.parent.papers)

        return
    
    def getAllPaperTerms(self):
        # all_terms = []
        # for paper_score, paper in self.papers:
        #     all_terms.extend(paper.node_terms[self.label])
        return self.all_paper_terms
    
    def sim_lexical(self):
        idf = lambda w: np.log(len(self.taxo.collection)/np.sum([1 for paper in self.taxo.collection if w in paper.vocabulary]))
        w_idf = {term:idf(term) for term in self.all_node_terms}

        sim_l = [0 for i in np.arange(len(self.taxo.collection))]
        for p_id, p in enumerate(self.taxo.collection):
            for term in self.all_node_terms:
                sim_l[p_id] += p.vocabulary[term] * w_idf[term]
        return sim_l/len(self.taxo.all_node_terms)
    
    def sim_semantic_phrase(self):
        
        return
    
    def rankPapers(self, class_reprs, phrase=True):
        paper_reprs = []
        for p in self.papers:
            paper_reprs.append(p.computePaperEmb(class_reprs))
        
        if phrase:
            new_paper_scores = cosine_similarity_embeddings(paper_reprs, [self.phrase_emb]).reshape((-1,))
        else:
            new_paper_scores = cosine_similarity_embeddings(paper_reprs, [self.emb]).reshape((-1,))

        self.paper_scores = {}
        for p_id, paper in enumerate(self.papers):
            self.paper_scores[paper.id] = new_paper_scores[p_id]

        self.papers = sorted(self.papers, key=lambda x: self.paper_scores[x.id], reverse=True)

        return self.papers


class Taxonomy:
    def __init__(self, track=None, dimen=None, input_file=None):
        self.collection = []
        self.raw_emb = None
        self.static_emb = None
        self.min_freq = 3
        
        if input_file is not None:
            id = 0
            with open(input_file, "r") as f:
                papers = f.read().strip().splitlines()
                for p in papers:
                    title = re.findall(r'title\s*:\s*(.*) ; abstract', p, re.IGNORECASE)[0]
                    abstract = re.findall(r'abstract\s*:\s*(.*)', p, re.IGNORECASE)[0]
                    self.collection.append(Paper(self, id, title, abstract, p))
                    id += 1
        
        self.root = Node(self, f"Types of {dimen} Proposed in {track} Research Papers")
        self.height = 0

    def __repr__(self) -> str:
        return json.dumps(self.toDict())

    def __str__(self) -> str:
        return json.dumps(self.toDict(), indent=2)
    
    def toDict(self, node=None, with_phrases=False):
        if node is None:
            node = self.root
        
        if not node:
            return {}
        
        if with_phrases:
            out = {node.label: {"description": node.desc, "seeds": node.seeds, "terms": node.all_node_terms}}
        else:
            out = {node.label: {"description": node.desc, "children": {}}}
        
        queue = deque([(node, out[node.label]["children"])])
        
        while queue:
            current_node, current_dict = queue.popleft()
            
            for child in current_node.children:
                if with_phrases:
                    current_dict[child.label] = {"description": child.desc, "seeds": child.seeds, "terms": child.all_node_terms}
                else:
                    current_dict[child.label] = {"description": child.desc, "children": {}}
                queue.append((child, current_dict[child.label]["children"]))
        
        return out
    
    def buildBaseTaxo(self, levels=2, k=5, num_terms=10):
        children = [self.root.genCommonSenseChildren(k=k, num_terms=num_terms)]
        self.height += 1

        for l in range(levels-1):
            children.append([])
            global_taxo = self.toDict()
            for child in children[l]:
                children[l+1].extend(child.genCommonSenseChildren(global_taxo, k, num_terms=num_terms))
            self.height += 1
        
        return self.toDict()
    
    def getClassReprs(self, class_nodes, phrase=True):
        class_reprs = []
        for cls in class_nodes:
            class_reprs.append(cls.updateNodeEmb(phrase))
        return class_reprs
    
    def mapPapers(self, paper_reprs, class_nodes, class_reprs):

        # no. of papers x no. of classes
        cos_sim = cosine_similarity_embeddings(paper_reprs, class_reprs)

        # identify lower-bound for each class (based on bottom-most ranked paper)
        lower_bounds = [cos_sim[c.papers[-1].id, c_id] for c_id, c in enumerate(class_nodes)]

        # get the classes for each paper which have a sim above the lower-bound threshold
        bottom_classes = np.argmax(np.diff(np.sort(cos_sim, axis=1), axis=1), axis=1) + 1

        classes = np.argsort(cos_sim, axis=1)
        class_labels = []
        mapping = {i:[] for i in np.arange(-1, len(class_nodes))}
        for p_id, b in enumerate(bottom_classes):
            class_labels.append([])
            for cls in classes[p_id][b:]:
                if cos_sim[p_id, cls] >= lower_bounds[cls]:
                    class_labels[p_id].append(cls)
                    class_nodes[cls].addPaper(self.collection[p_id])
                    mapping[cls].append(p_id)
                
            if len(class_labels[p_id]) == 0:
                mapping[-1].append(p_id)

        return class_labels, mapping
    
    def siblingExpansion(self, parent_node, mapping):
        # get non-stopword oov vocab from unmapped papers
        unmapped_vocab = set()
        for p_id in mapping[-1]:
            unmapped_vocab.update([w for w in self.collection[p_id].vocabulary 
                                   if (w not in stopwords.words('english') and (w not in parent_node.all_node_terms))])
        unmapped_vocab = list(unmapped_vocab)

        phrase_reprs = [self.static_emb[w] for w in unmapped_vocab]
        # terms should be relevant to the parent node
        parent_ranks = rank_by_significance(phrase_reprs, [parent_node.phrase_emb])
        # terms should be dissimilar to terms from existing sibling nodes
        # sib_ranks = rank_by_insignificance(phrase_reprs, [c.phrase_emb for c in parent_node.children])

        # they should be frequent in the unmapped pool of papers but infrequent in the mapped pool of papers
        unmapped = []
        mapped = []
        for p in self.collection:
            if p.id in mapping[-1]:
                unmapped.append(p)
            else:
                mapped.append(p)
        lexical_ranks = rank_by_lexical(unmapped_vocab, mapped, unmapped)

        # compute joint rank
        # rankings = [parent_ranks, sib_ranks, lexical_ranks]
        rankings = [parent_ranks, lexical_ranks]
        rankings_len = len(unmapped_vocab)

        total_score = []
        for i in range(rankings_len):
            total_score.append(mul(ranking[i] for ranking in rankings))

        total_ranking = {unmapped_vocab[i]: r for r, i in enumerate(np.argsort(np.array(total_score)))}

        return total_ranking