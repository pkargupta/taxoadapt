# analysis.py

from utils import (
    get_root,
    get_taxonomy,
    get_paths,
    get_levels,
    get_all_nodes,
    node_name2titles,
)

def analyze_json(eval_json: dict):
    """
    Process the evaluation JSON to extract claim, taxonomy, paths, levels, and nodes.
    """
    root = get_root(eval_json)
    print("Get the root.")
    
    taxonomy = get_taxonomy(eval_json)
    print("Get 1 taxonomy.")
    
    paths = get_paths(eval_json)
    # remove the root from each path for clarity
    for i in range(len(paths)):
        paths[i] = paths[i].replace(root + " -> ", "")
    print(f"Get {len(paths)} path(s).")
    
    levels = get_levels(eval_json)
    print(f"Get {len(levels)} level(s).")

    raw_nodes = get_all_nodes(eval_json)
    print(f"Get {len(raw_nodes)} node(s).")
    
    nodes = [node_name2titles(node) for node in raw_nodes]
    
    return root, taxonomy, paths, levels, nodes
