import os
import re
from collections import deque
import json
from model_definitions import llama_8b_model
from prompts import depth_expansion_init_prompt, depth_expansion_prompt


class Paper:
    def __init__(self, id, title, abstract):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.text = f"title: {title}; abstract: {abstract}"
        
        self.sentences = []
        self.terms = []

        self.nodes = [] # list of tuples (score, label)
        self.node_terms = {} # key: node label; value: paper-specific terms relevant to node

    def __repr__(self) -> str:
        return self.text

class Node:
    def __init__(self, label, seeds=None, description=None, parent=None):
        self.label = label
        self.model = llama_8b_model

        self.parent = parent
        self.children = []

        self.seeds = seeds
        self.desc = description

        self.papers = [] # list of tuples (score, paper)
        self.density = 0

        self.all_node_terms = [s.lower().replace(" ", "_") for s in seeds] if seeds is not None else []
        self.all_paper_terms = []

    def __repr__(self) -> str:
        return self.label

    def getChildren(self, terms=False):
        if terms:
            return [(c.label, c.all_node_terms) for c in self.children]
        else:
            return [c.label for c in self.children]
    
    def addChild(self, label, seeds, desc):
        child_node = Node(label, seeds, desc, parent=self)
        self.children.append(child_node)
        return child_node
    
    def addChildren(self, labels, seeds, descriptions):
        for l, s, d in zip(labels, seeds, descriptions):
            child_node = Node(l, s, d, parent=self)
            self.children.append(child_node)
            self.addTerms(s, True)

        return self.children
    
    def addTerms(self, terms, addToParent=False):
        for t in terms:
            mod_t = t.lower().replace(" ", "_")
            if mod_t not in self.all_node_terms:
                self.all_node_terms.append(mod_t)

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
    def __init__(self, track=None, dimen=None):
        self.root = Node(f"Types of {dimen} Proposed in {track} Research Papers")
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
        