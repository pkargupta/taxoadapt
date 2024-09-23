import os
import re
import numpy as np
from collections import defaultdict, deque, Counter
import itertools
from tqdm import tqdm
import pickle as pk
from nltk.corpus import stopwords
import torch
from model_definitions import sentence_model, bert_model, bert_tokenizer, bertEncode, chunkify, get_vocab_idx, get_hidden_states
from utils import rank_by_discriminative_significance, weights_from_ranking, cosine_similarity_embeddings, rank_by_significance, rank_by_class_discriminative_significance

class Paper:
    def __init__(self, taxo, raw_text, title, abstract, content=None, id=-1, gold=[]) -> None:
        self.taxo = taxo
        self.id = id
        self.raw_text = raw_text
        self.title = title
        self.abstract = abstract
        self.content = content
        self.gold = gold

        self.sent_tokenize = taxo.sent_tokenize[self.id]
        self.phrase_tokenize = taxo.phrase_tokenize[self.id]
        self.vocabulary = dict(Counter(itertools.chain.from_iterable(self.phrase_tokenize)))


    def __repr__(self) -> str:
        return f"id: {self.id}; title: {self.title}; abstract: {self.abstract}"
    

    def rankPhrases(self, class_reprs):
        in_vocab = list(self.vocabulary.keys())
        phrase_reprs = np.concatenate([self.taxo.vocab['phrases'][w].reshape((-1, 768)) for w in in_vocab], axis=0)
        ranks = rank_by_discriminative_significance(phrase_reprs, class_reprs)
        ranked_tok = {in_vocab[idx]:rank for idx, rank in ranks.items()}
        return ranked_tok
    
    def rankSentences(self, class_reprs, phrases=True):
        if phrases:
            ranked_phrases = self.rankPhrases(class_reprs)

            sent_avg_weights = []
            sent_reprs = []
            for sent in self.sent_tokenize:
                phrase_reprs = []
                phrase_ranks = {}
                p_id = 0
                for phrase in sent.split():
                    if phrase not in stopwords.words('english'):
                        phrase_reprs.append(self.taxo.static_emb[phrase])
                        phrase_ranks[p_id] = ranked_phrases[phrase]
                        p_id += 1
                
                if len(phrase_ranks) > 0:
                    phrase_weights = weights_from_ranking({k: v for k, v in sorted(phrase_ranks.items(), key=lambda item: item[1])})
                    sent_avg_weights.append(np.mean(list(phrase_ranks.values())))

                    # sent_reprs.append(sentence_model.encode(" ".join(sent)))
                    sent_reprs.append(np.average(phrase_reprs, weights=phrase_weights, axis=0))
            
            sent_avg_ranks = {idx:rank for rank, idx in enumerate(np.argsort(sent_avg_weights))}
        else:
            sent_reprs = [self.taxo.vocab['sentences'][sent] for sent in self.sent_tokenize]
            sent_avg_ranks = None
        
        sent_dis_ranks = rank_by_discriminative_significance(sent_reprs, class_reprs)
        return sent_reprs, sent_avg_ranks, sent_dis_ranks
    
    def computePaperEmb(self, class_reprs=None, phrase=False):
        sent_reprs, sent_avg_ranks, sent_dis_ranks = self.rankSentences(class_reprs, phrase)
        if sent_avg_ranks is not None:
            weights = weights_from_ranking([sent_avg_ranks, sent_dis_ranks])
        else:
            weights = weights_from_ranking([sent_dis_ranks])

        self.emb = np.average(sent_reprs, weights=weights, axis=0)
        return self.emb

class Node:
    def __init__(self, node_id, label, description=None, level=0):
        self.label = label
        self.level = level
        self.description = description #f'{self.label} : {description}' # optional
        self.node_id = node_id
        self.parents = []
        self.children = []
        self.path = [self.label]
        self.similarity_score = 0
        self.path_score = 0

        # enrichment
        self.common_sense = {"phrases": [], "sentences": [], "examples": []}
        self.external = {"phrases": [], "sentences": [], "examples": []}
        self.internal = {"phrases": [], "sentences": [], "examples": []}
        self.all = {"phrases": [], "sentences": [], "examples": []}

        # classification
        self.emb = {} # 'phrase':emb, 'sentence':emb
        self.gold = {}
        self.papers = {} # id: Paper

    def __repr__(self) -> str:
        return self.label
    
    def resetNode(self):
        self.external = {"phrases": [], "sentences": [], "examples": []}
        self.internal = {"phrases": [], "sentences": [], "examples": [], "sent_ids": []}
        self.all = {"phrases": [], "sentences": [], "examples": []}

        # classification
        self.emb = {}
        if self.node_id != "0":
            self.papers = {} # id: Paper

    def addChild(self, child):
        if child not in self.children:
            self.children.append(child)

    def addParent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)
            self.path = parent.path + self.path

    def findChild(self, node_id):
        if type(node_id) == int:
            node_id = str(node_id)
        if node_id == self.node_id:
            return self
        if len(self.children) == 0:
            return None
        for child in self.children:
            ans = child.findChild(node_id)
            if ans != None:
                return ans
        return None

    def getAllTerms(self, children=True, granularity='phrases'):
        # TODO: SET THIS BACK!!
        # all_node_terms = [self.label if granularity == 'phrases' else self.description]
        all_node_terms = []
        all_node_terms.extend([w for w in self.common_sense[granularity] if w not in all_node_terms])
        all_node_terms.extend([w for w in self.external[granularity] if w not in all_node_terms])
        all_node_terms.extend([w for w in self.internal[granularity] if w not in all_node_terms])

        if children:
            for c in self.children:
                all_node_terms.extend(c.getAllTerms(children=True, granularity=granularity))

        return list(all_node_terms)

    def getChildren(self, terms=False):
        if terms:
            return [(c.label, c.getAllTerms()) for c in self.children]
        else:
            return [c.label for c in self.children]

class Taxonomy:
    def __init__(self, data_dir):
        self.collection = []
        self.external_collection = []
        # sentence_model embeddings
        self.vocab = {'phrases':{},
                      'sentences': {},
                      'examples': {}}
        
        self.static_emb = {'phrases':{},
                           'sentences': {},
                           'examples': {}}
        self.vocab_count = defaultdict(int)
        self.token_lens = {}
        self.sent_tokenize = {} # paper id: [[tokenized sent_1] ... [tokenized sent_k]]
        self.phrase_tokenize = {} # paper id: [tokenized doc]
        
        self.graph, self.id2label, self.label2id = self.createGraph(os.path.join(data_dir), 'labels.txt')
        self.root = self.graph.findChild('0')

        self.gold_labels = []
    

    # create taxonomy from input file
    def createGraph(self, file_addr, label_file='labels.txt'):
        root = None
        # sanity check if file exist
        if not os.path.exists(file_addr):
            print(f"ERROR. Taxonomy file addr {file_addr} not exists.")
            exit(-1)

        id2label = {}
        id2desc = {}
        label2id = {}

        with open(os.path.join(file_addr, label_file)) as f:
            for line in f:
                line_info = line.strip().split('\t')
                # without description
                if len(line_info) == 2:
                    label_id, label_name = line_info
                # with description
                if len(line_info) == 3:
                    label_id, label_name, label_desc = line_info
                    id2desc[label_id] = label_desc

                id2label[label_id] = label_name
                label2id[label_name] = label_id

        # construct graph from file
        with open(os.path.join(file_addr, 'label_hierarchy.txt')) as f:
            ## for each line in the file
            root = Node(-1, 'ROOT')
            for line in f:
                parent_id, child_id = line.strip().split('\t')
                parent = id2label[parent_id]
                child = id2label[child_id]
                parent_desc = id2desc[parent_id] if len(id2desc) > 0 else None
                child_desc = id2desc[child_id] if len(id2desc) > 0 else None

                parent_node = root.findChild(parent_id)
                if parent_node is None:
                    parent_node = Node(parent_id, parent, description=parent_desc, level=1)
                    if parent_node.label not in self.vocab['phrases']:
                        self.vocab['phrases'][parent_node.label] = sentence_model.encode(parent_node.label)
                        if parent_node.description:
                            self.vocab['sentences'][parent_node.description] = sentence_model.encode(parent_node.description)
                            self.vocab['sentences'][f'{parent_node.label}: {parent_node.description}'] = sentence_model.encode(f'{parent_node.label}: {parent_node.description}')

                    root.addChild(parent_node)
                    parent_node.addParent(root)

                child_node = root.findChild(child_id)
                if child_node is None:
                    child_node = Node(child_id, child, description=child_desc, level=parent_node.level+1)
                    if child_node.label not in self.vocab['phrases']:
                        self.vocab['phrases'][child] = sentence_model.encode(child_node.label)
                        if child_node.description:
                            self.vocab['sentences'][child_node.description] = sentence_model.encode(child_node.description)
                            self.vocab['sentences'][f'{child_node.label}: {child_node.description}'] = sentence_model.encode(f'{child_node.label}: {child_node.description}')
                    
                parent_node.addChild(child_node)
                child_node.addParent(parent_node)
        
        return root, id2label, label2id
    
    def resetTaxo(self):
        # this function clears the enrichment and classification steps for all nodes

        queue = deque([self.root])

        while queue:
            curr_node = queue.popleft()
            curr_node.resetNode()
            for child in curr_node.children:
                queue.append(child)

    def toDict(self, cur_node=None, idx=0):
        # turns the subtree for this node (inclusive) into a dictionary
        gen_dict = lambda node: {"id": node.node_id, 
                                   "description": node.description, 
                                   "example_key_phrases": node.common_sense['phrases'], 
                                   "example_sentences": node.common_sense['sentences'], 
                                   "children": {}
                                   } if len(node.common_sense['phrases']) > 0 else {
                                       "id": node.node_id,
                                       "description": node.description,
                                       "children": {}
                                   }

        if cur_node is None:
            cur_node = self.root.findChild(idx)
            out = {cur_node.label:gen_dict(cur_node)}
        else: # root
            out = {cur_node.label:{"id": cur_node.node_id, "description": cur_node.description, "children": {}}}

        queue = deque([(cur_node, out[cur_node.label]["children"])])

        while queue:
            current_node, current_dict = queue.popleft()
            for child in current_node.children:
                current_dict[child.label] = gen_dict(child)
                queue.append((child, current_dict[child.label]["children"]))

        return out

    def get_sib(self, idx, granularity='phrases'):
        cur_node = self.root.findChild(idx)
        par_node = cur_node.parents
        sib = []
        for j in par_node:
            sib.extend([i for i in j.children if i != cur_node])
        
        if granularity == 'phrases':
            return [i.label for i in sib]
        elif granularity == 'sentences':
            return [i.description for i in sib]
        else:
            return sib

    def get_par(self, idx):
        cur_node = self.root.findChild(idx)
        parent_list = []

        for par_node in cur_node.parents:
            if par_node != self.root:
                parent_list.append(par_node.label)
                for par_par_node in par_node.parents:
                    if par_par_node != self.root:
                        parent_list.append(par_par_node.label)
        return [i.replace("_"," ") for i in parent_list]
    
    def addPapertoCollection(self, paper_idx, paper, external=False):
        if paper_idx < len(self.collection) + len(self.external_collection):
            return
        else:
            title = re.findall(r'paper_title\s*:\s*(.*) ; paper_abstract', paper, re.IGNORECASE)[0]
            abstract = re.findall(r'paper_abstract\s*:\s*(.*)', paper, re.IGNORECASE)[0] if external else re.findall(r'paper_abstract\s*:\s*(.*) ; paper_content : ', paper, re.IGNORECASE)[0]
            content = None if external else re.findall(r'paper_content\s*:\s*(.*)', paper, re.IGNORECASE)[0]
            if external:
                self.external_collection.append(Paper(self, paper, title, abstract, content, paper_idx))
            else:
                paper_node = Paper(self, paper, title, abstract, content, paper_idx, gold=self.gold_labels[paper_idx])
                self.collection.append(paper_node)
                self.root.papers[paper_idx] = paper_node
    
    def computeStaticEmb(self, args):
        if os.path.exists(os.path.join(args.data_dir, 'static_emb.pk')):
            print("Loading in static embeddings and updating...")
            with open(os.path.join(args.data_dir, 'static_emb.pk'), "rb") as f:
                saved_emb = pk.load(f)
                self.static_emb = saved_emb["static_emb"]
            
            unseen_terms = []
            for paper in tqdm(self.collection + self.external_collection):
                chunks = chunkify(paper.raw_text, self.token_lens, args.length)
                for chunk in chunks:
                    unseen = [word for word in chunk if word not in self.static_emb]
                    if len(unseen):
                        unseen_terms.extend(unseen)
                        for word in unseen:
                            self.static_emb[word] = np.zeros((args.dim,))
                        bertEncode(chunk, self)
            
            for term in unseen_terms:
                self.static_emb[term] = self.static_emb[term]/self.vocab_count[term]
            
            with open(os.path.join(args.data_dir, 'static_emb.pk'), "wb") as f:
                pk.dump({"static_emb": self.static_emb}, f)
            
        else:
            for paper in tqdm(self.collection + self.external_collection):
                chunks = chunkify(paper.raw_text, self.token_lens, args.length)
                for chunk in chunks:
                    bertEncode(chunk, self)

            self.static_emb = {w:w_emb/self.vocab_count[w] for w, w_emb in self.static_emb.items()}

            with open(os.path.join(args.data_dir, 'static_emb.pk'), "wb") as f:
                pk.dump({"static_emb": self.static_emb}, f)
        
    
    def createCollections(self, args):
        # get vocab freq + token_lens
        self.static_emb = {}
        print("Constructing Collections...")
        with open(os.path.join(args.data_dir, args.groundtruth), "r") as f:
            self.gold_labels = [line.split(",") for line in f.read().strip().splitlines()]
        
        with open(os.path.join(args.data_dir, "phrase_" + args.internal)) as fin:
            papers = [l.strip() for l in fin]
            len_internal = len(papers)
        with open(os.path.join(args.data_dir, "phrase_" + args.external)) as fin:
            papers.extend([l.strip() for l in fin])

        for paper_idx, paper in tqdm(enumerate(papers), total=len(papers)):
            sents = paper.split(" . ")
            phrases = [sent.split() for sent in sents]
            self.sent_tokenize[paper_idx] = sents
            self.phrase_tokenize[paper_idx] = phrases
            data = paper.split()

            if paper_idx < len_internal:
                self.addPapertoCollection(paper_idx, paper, external=False)
            else:
                self.addPapertoCollection(paper_idx, paper, external=True)

            for word in data:
                self.vocab_count[word] += 1
                if word not in self.token_lens:
                    self.token_lens[word] = bert_tokenizer.tokenize(word.replace("_", " "))
                    self.static_emb[word] = np.zeros((args.dim, ))

        # adding gold labels to nodes
        gold_map = {}
        for p_id, paper in enumerate(self.gold_labels):
            for node_id in paper:
                if node_id in gold_map:
                    gold_map[node_id].append(p_id)
                else:
                    gold_map[node_id] = [p_id]
        
        for node_id in gold_map.keys():
            l_node = self.root.findChild(node_id)
            for paper_id in gold_map[node_id]:
                l_node.gold[paper_id] = self.collection[paper_id]
        
        print("Computing Static Embeddings...")
        self.computeStaticEmb(args)

        return self.collection, self.external_collection
    
    def updateVocab(self, text, granularity='phrases'):
        if type(text) == list:
            filtered_text = [text_item for text_item in text if not (text_item in self.vocab[granularity])]
            if len(filtered_text) > 0:
                embs = sentence_model.encode(filtered_text)
                self.vocab[granularity].update({text_item:emb for text_item, emb in zip(filtered_text, embs)})
                return None
        else:
            if not (text in self.vocab[granularity]):
                self.vocab[granularity][text] = sentence_model.encode(text)
            
            return self.vocab[granularity][text]
    
    def rankPapers(self, class_reprs, class_id, internal=True, phrase=False, top_k=20):
        if internal:
            paper_embs = [p.computePaperEmb(class_reprs, phrase) for p in self.collection]
        else:
            paper_embs = [p.computePaperEmb(class_reprs, phrase) for p in self.external_collection]
        if len(class_reprs) == 1:
            ranks = rank_by_significance(paper_embs, class_reprs)
        else:
            ranks = rank_by_class_discriminative_significance(paper_embs, class_reprs, class_id)
        
        sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
        return sorted_ranks[:top_k]
    
    def mapPapers(self, paper_reprs, class_nodes, class_reprs):

        # no. of papers x no. of classes
        cos_sim = cosine_similarity_embeddings(paper_reprs, class_reprs)

        # identify lower-bound for each class (based on bottom-most ranked paper)
        lower_bounds = [cos_sim[-1, c_id] for c_id, c in enumerate(class_nodes)]

        # for each paper, only consider the classes that are above the largest similarity gap
        bottom_classes = np.argmax(np.diff(np.sort(cos_sim, axis=1), axis=1), axis=1) + 1

        classes = np.argsort(cos_sim, axis=1)
        class_labels = []
        class_paper_scores = []
        mapping = {i:[] for i in np.arange(-1, len(class_reprs))}
        for p_id, b in enumerate(bottom_classes):
            class_labels.append([])
            class_paper_scores.append([])

            for cls in classes[p_id][b:]:
                # get the classes for each paper which have a sim above the lower-bound threshold
                if cos_sim[p_id, cls] >= lower_bounds[cls]:
                    class_labels[p_id].append(cls)
                    class_paper_scores[p_id].append(cos_sim[p_id, cls])
                    # class_nodes[cls].addPaper(class_nodes[0].parent.papers[p_id])
                    # class_nodes[cls].addPaper(self.collection[p_id])

                    # update paper score
                    # class_nodes[cls].paper_scores[p_id] = cos_sim[p_id, cls]

                    mapping[cls].append(p_id)
                
            if len(class_labels[p_id]) == 0:
                mapping[-1].append(p_id)
        
        # re-sort papers
        # for cls in class_nodes:
        #     cls.papers = sorted(cls.papers, key=lambda x: cls.paper_scores[x.id], reverse=True)

        return class_labels, class_paper_scores, mapping
