import os
import re
import numpy as np
from collections import deque, Counter
import json
from model_definitions import llama_8b_model, sentence_model
from prompts import depth_expansion_init_prompt, depth_expansion_prompt
from utils import average_with_harmonic_series, cosine_similarity_embeddings
from utils import rank_by_significance, rank_by_discriminative_significance, weights_from_ranking


class Paper:
    def __init__(self, taxo, id, title, abstract, text):
        self.taxo = taxo
        self.id = id
        self.title = title
        self.abstract = abstract
        self.text = text # f"title : {title}; abstract: {abstract}"
        self.split_text = self.text.split(" ")
        self.length = len(self.split_text)
        
        
        self.sentences = []
        self.terms = []

        self.nodes = {} # path: score
        self.node_terms = {} # key: node label; value: paper-specific terms relevant to node

        self.vocabulary = dict(Counter(self.split_text))

        self.emb = None

    def __repr__(self) -> str:
        return self.text
    
    def addNodeTerms(self, node, terms):
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
        iv_phrases = [w for w in self.vocabulary if w in self.taxo.word2emb]
        phrase_reprs = np.concatenate([self.taxo.word2emb[w].reshape((-1, 768)) for w in iv_phrases], axis=0)
        ranks = rank_by_significance(phrase_reprs, class_reprs)
        ranked_tok = {iv_phrases[idx]:rank for idx, rank in ranks.items()}
        return ranked_tok
    
    def rankSentences(self, class_reprs):
        ranked_phrases = self.rankPhrases(class_reprs)

        sent_avg_weights = []
        sent_reprs = []
        for sent in self.sentences:
            phrase_reprs = []
            phrase_ranks = {}
            p_id = 0
            for phrase in sent:
                if phrase in self.taxo.word2emb:
                    phrase_reprs.append(self.taxo.word2emb[phrase])
                    phrase_ranks[p_id] = ranked_phrases[phrase]
                    p_id += 1
            
            if len(phrase_ranks) > 0:
                phrase_weights = weights_from_ranking({k: v for k, v in sorted(phrase_ranks.items(), key=lambda item: item[1])})
                sent_avg_weights.append(np.mean(list(phrase_ranks.values())))
                # sent_reprs.append(sentence_model.encode(" ".join(sent)))
                sent_reprs.append(np.average(phrase_reprs, weights=phrase_weights, axis=0))
        
        sent_ranks = {idx:rank for rank, idx in enumerate(np.argsort(sent_avg_weights))}

        return sent_reprs, sent_ranks
    
    def computePaperEmb(self, class_reprs=None):
        # SPECTER-based
        # self.emb = sentence_model.encode(self.title + "[SEP]" + self.abstract)

        # sent-based repr
        sent_reprs, sent_ranks = self.rankSentences(class_reprs)
        weights = weights_from_ranking(sent_ranks)
        self.emb = np.average(sent_reprs, weights=weights, axis=0)

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

        self.collection = self.taxo.collection
        self.papers = [] # list of tuples (score, paper)
        self.density = 0

        self.mined_terms = []
        self.all_node_terms = [s.lower().replace(" ", "_") for s in [self.label] + seeds]

        # get initial pool of papers
        for paper in self.collection:
            freq = paper.addNodeTerms(self, self.all_node_terms)
            if freq >= self.taxo.min_freq:
                self.papers.append((freq, paper))

        self.papers = sorted(self.papers, key=lambda x: x[0], reverse=True)
        
        self.all_paper_terms = []
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
            self.emb = average_with_harmonic_series([sentence_model.encode(p[1].title + "[SEP]" + p[1].abstract) 
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

        for paper in self.collection:
            freq = paper.addNodeTerms(self, new_terms)
            if freq >= self.taxo.min_freq: # min frequency
                self.papers.append((freq, paper))

        self.papers = sorted(self.papers, key=lambda x: x[0], reverse=True)

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
        self.papers.append(paper)
        if self.label in paper.node_terms.keys():
            self.all_paper_terms.extend(paper.node_terms[self.label])
        # update density (TODO: open problem):
        self.density = len(self.papers)/len(self.parent.papers)

        return
    
    def getAllPaperTerms(self):
        # all_terms = []
        # for paper_score, paper in self.papers:
        #     all_terms.extend(paper.node_terms[self.label])
        return self.all_paper_terms
    
    def rankPapers(self, class_reprs, phrase=True):
        paper_reprs = []
        for p in self.papers:
            paper_reprs.append(p[1].computePaperEmb(class_reprs))
        
        if phrase:
            paper_scores = cosine_similarity_embeddings(paper_reprs, [self.phrase_emb]).reshape((-1,))
        else:
            paper_scores = cosine_similarity_embeddings(paper_reprs, [self.emb]).reshape((-1,))

        papers = [(paper_scores[i], self.papers[i][1]) for i in np.arange(len(self.papers))]

        self.papers = sorted(papers, key=lambda x: x[0], reverse=True)

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
    
    def getClassReprs(self, class_nodes):
        class_reprs, class_phrase_reprs = [], []
        for cls in class_nodes:
            class_reprs.append(cls.updateNodeEmb(phrase=False))
            class_phrase_reprs.append(cls.updateNodeEmb(phrase=True))
        return class_reprs, class_phrase_reprs
    
    def mapPapers(self, paper_reprs, class_reprs):
        # no. of papers x no. of classes
        cos_sim = cosine_similarity_embeddings(paper_reprs, class_reprs)
        bottom_classes = np.argmax(np.diff(np.sort(cos_sim, axis=1), axis=1), axis=1) + 1

        classes = np.argsort(cos_sim, axis=1)
        class_labels = [] 
        for p_id, b in enumerate(bottom_classes):
            class_labels.append(classes[p_id][b:])

        return class_labels