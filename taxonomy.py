import os
import re
from collections import deque

class Paper:
    def __init__(self, id, title, abstract):
        self.id = id
        self.title = title
        self.abstract = abstract
        self.text = f"title: {title}; abstract: {abstract}"
        
        self.sentences = []
        self.terms = []

        self.nodes = [] # list of tuples (score, label)

class Node:
    def __init__(self, label, seeds=None, description=None, parent=None):
        self.label = label

        self.parent = parent
        self.children = []

        self.seeds = seeds
        self.desc = description

        self.papers = [] # list of tuples (score, paper)
        self.density = 0
    
    def addChild(self, label, seeds, desc):
        child_node = Node(label, seeds, desc, parent=self)
        self.children.append(child_node)
        return child_node
    
    def addChildren(self, labels, seeds, descriptions):
        for l, s, d in zip(labels, seeds, descriptions):
            child_node = Node(l, s, d, parent=self)
            self.children.append(child_node)
        
        return self.children
    
    def addPaper(self, paper):
        self.papers.append(paper)
        
        # update density:
        self.density += 1

        return

class Taxonomy:
    def __init__(self, track=None, dimen=None):
        self.root = Node(track)
        self.dimension = dimen
    
    def toDict(self, node=None):
        if node is None:
            node = self.root
        
        if not node:
            return {}
        
        out = {node.label: {}}
        queue = deque([(node, out[node.label])])
        
        while queue:
            current_node, current_dict = queue.popleft()
            
            for child in current_node.children:
                current_dict[child.label] = {}
                queue.append((child, current_dict[child.label]))
        
        return out