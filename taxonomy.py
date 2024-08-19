import os
import re
from tqdm import tqdm
from collections import deque

class Node:
    def __init__(self, node_id, label, description=None, level=0):
        self.label = label
        self.level = level
        self.description = description # optional
        self.node_id = node_id
        self.parents = []
        self.children = []
        self.similarity_score = 0
        self.path_score = 0

        # enrichment
        self.common_sense = {"phrases": [], "sentences": [], "examples": []}
        self.external = {"phrases": [], "sentences": [], "examples": []}
        self.corpus = {"phrases": [], "sentences": [], "examples": []}

        # classification
        self.papers = {} # id: Paper

    def __repr__(self) -> str:
        return self.label

    def addChild(self, child):
        if child not in self.children:
            self.children.append(child)

    def addParent(self, parent):
        if parent not in self.parents:
            self.parents.append(parent)

    def findChild(self, node_id):
        if node_id == self.node_id:
            return self
        if len(self.children) == 0:
            return None
        for child in self.children:
            ans = child.findChild(node_id)
            if ans != None:
                return ans
        return None

class Taxonomy:
    def __init__(self, root:Node):
        self.root = root.findChild('0')

    def toDict(self, cur_node=None, idx=0):
        # turns the subtree for this node (inclusive) into a dictionary
        if cur_node is None:
            cur_node = self.root.findChild(idx)
            out = {cur_node.label:{"id": cur_node.node_id, "description": cur_node.description, "example_key_phrases": [], "example_sentences": [], "children": {}}}
        else: # root
            out = {cur_node.label:{"id": cur_node.node_id, "children": {}}}

        queue = deque([(cur_node, out[cur_node.label]["children"])])

        while queue:
            current_node, current_dict = queue.popleft()
            for child in current_node.children:
                current_dict[child.label] = {"id": child.node_id, "description": child.description, "example_key_phrases": [], "example_sentences": [], "children": {}}

                queue.append((child, current_dict[child.label]["children"]))

        return out

    def get_sib(self, idx):
        cur_node = self.root.findChild(idx)
        par_node = cur_node.parents
        sib = []
        for j in par_node:
            sib.extend([i for i in j.children if i != cur_node])
        return [i.label.replace("_"," ") for i in sib]

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
