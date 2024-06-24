import os
import re
import numpy as np
from collections import deque, Counter
import json
from model_definitions import llama_8b_model
from prompts import depth_expansion_init_prompt, depth_expansion_prompt


class Paper:
    def __init__(self, id, title, abstract):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.text = f"title: {title}; abstract: {abstract}"
        self.split_text = self.text.split(" ")
        self.length = len(self.split_text)
        
        
        self.sentences = []
        self.terms = []

        self.nodes = {} # path: score
        self.node_terms = {} # key: node label; value: paper-specific terms relevant to node

        self.vocabulary = dict(Counter(self.split_text))

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


class Node:
    def __init__(self, label, seeds=None, description=None, parent=None, collection=[]):
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

        self.collection = collection
        self.papers = [] # list of tuples (score, paper)
        self.density = 0

        self.all_node_terms = [s.lower().replace(" ", "_") for s in seeds] if seeds is not None else []
        for paper in self.collection:
            freq = paper.addNodeTerms(self, self.all_node_terms)
            if freq > 0:
                self.papers.append((freq, paper))
        
        self.all_paper_terms = []

    def __repr__(self) -> str:
        return self.label
    
    def getPath(self):
        return self.path

    def getChildren(self, terms=False):
        if terms:
            return [(c.label, c.all_node_terms) for c in self.children]
        else:
            return [c.label for c in self.children]
    
    def addChild(self, label, seeds, desc):
        child_node = Node(label, seeds, desc, parent=self, collection=self.collection)
        self.children.append(child_node)
        return child_node
    
    def addChildren(self, labels, seeds, descriptions):
        for l, s, d in zip(labels, seeds, descriptions):
            child_node = Node(l, s, d, parent=self, collection=self.collection)
            self.children.append(child_node)
            self.addTerms(s, True)

        return self.children
    
    def addTerms(self, terms, addToParent=False):
        new_terms = []
        for t in terms:
            mod_t = t.lower().replace(" ", "_")
            if mod_t not in self.all_node_terms:
                self.all_node_terms.append(mod_t)
                new_terms.append(mod_t)

        for paper in self.collection:
            freq = paper.addNodeTerms(self, new_terms)
            if freq > 0:
                self.papers.append((freq, paper))

        if addToParent and (self.parent is not None):
            self.parent.addTerms(terms, addToParent)

    
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
        labels = re.findall(r'label:\s*(.*)', message, re.IGNORECASE)
        descriptions = re.findall(r'description:\s*(.*)', message, re.IGNORECASE)
        seeds = [i.split(", ") for i in re.findall(r'terms:\s*\[(.*)\]', message, re.IGNORECASE)]

        return self.addChildren(labels, seeds, descriptions)
    
    def addPaper(self, paper):
        self.papers.append(paper)
        if self.label in paper.node_terms.keys():
            self.all_paper_terms.extend(paper.node_terms[self.label])
        # update density (TODO: open problem):
        self.density += 1

        return
    
    def getAllPaperTerms(self):
        # all_terms = []
        # for paper_score, paper in self.papers:
        #     all_terms.extend(paper.node_terms[self.label])
        return self.all_paper_terms

class Taxonomy:
    def __init__(self, track=None, dimen=None, input_file=None):
        self.collection = []
        if input_file is not None:
            id = 0
            with open(input_file, "r") as f:
                papers = f.read().strip().splitlines()
                for p in papers:
                    title = re.findall(r'title\s*:\s*(.*) ; abstract', p, re.IGNORECASE)[0]
                    abstract = re.findall(r'abstract\s*:\s*(.*)', p, re.IGNORECASE)[0]
                    self.collection.append(Paper(id, title, abstract))
                    id += 1
        
        self.root = Node(f"Types of {dimen} Proposed in {track} Research Papers", collection=self.collection)
        self.height = 0

    def __repr__(self) -> str:
        return json.dumps(self.toDict())

    def __str__(self) -> str:
        return json.dumps(self.toDict(), indent=2)
    
    def toDict(self, node=None):
        if node is None:
            node = self.root
        
        if not node:
            return {}
        
        out = {node.label: {"description": node.desc, "seeds": node.seeds, "terms": node.all_node_terms}}
        queue = deque([(node, out[node.label])])
        
        while queue:
            current_node, current_dict = queue.popleft()
            
            for child in current_node.children:
                current_dict[child.label] = {"description": child.desc, "seeds": child.seeds, "terms": child.all_node_terms}
                queue.append((child, current_dict[child.label]))
        
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
        