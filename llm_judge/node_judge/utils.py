# utils.py
import json

def get_json_files(eval_json_path: str) -> dict:
    """Load and return the JSON content from the specified file."""
    with open(eval_json_path, 'r') as f:
        return json.load(f)

# taxonomy.py

def get_root(eval_json: dict) -> str:
    return eval_json['label']

def get_aspect(eval_json: dict) -> str:
    return eval_json['label']

def get_taxonomy(taxonomy_json: dict, level: int = 0) -> str:
    expression = "  " * level + f"- {taxonomy_json['label']}\n"
    if 'children' in taxonomy_json:
        for child in taxonomy_json['children']:
            expression += get_taxonomy(child, level + 1)
    return expression

def present_taxonomy(taxonomy_json: dict, level: int = 0) -> str:
    """Recursively returns a string presentation of the taxonomy tree."""
    expression = "  " * level + f"- {taxonomy_json['label']}\n"
    if 'children' in taxonomy_json:
        for child in taxonomy_json['children']:
            expression += present_taxonomy(child, level + 1)
    return expression

def get_paths(node: dict, current_path: list = None) -> list:
    """
    Recursively traverse the tree and return a list of paths (strings) for each leaf node.
    """
    if current_path is None:
        current_path = []
    new_path = current_path + [node["label"]]
    if "children" not in node or not node["children"]:
        return [" -> ".join(new_path)]
    paths = []
    for child in node["children"]:
        paths.extend(get_paths(child, new_path))
    return paths

def get_levels(node) -> list:
    """
    Recursively traverse the tree and collect a list of dictionaries, each containing:
      - 'parent': the parent node’s aspect_name
      - 'siblings': a list of aspect_names of all children of that parent.
    """
    result = []
    if isinstance(node, dict):
        if 'children' in node and isinstance(node['children'], list) and len(node['children']) > 0:
            parent_name = node.get('label')
            siblings = [child.get('label') for child in node['children'] if 'label' in child]
            result.append({"parent": parent_name, "siblings": siblings})
            for child in node['children']:
                result.extend(get_levels(child))
        else:
            for value in node.values():
                if isinstance(value, (dict, list)):
                    result.extend(get_levels(value))
    elif isinstance(node, list):
        for item in node:
            result.extend(get_levels(item))
    return result

def get_all_nodes(tree: dict) -> list:
    """Return a list of all nodes in the taxonomy tree."""
    nodes = []
    def traverse(node):
        nodes.append(node)
        if "children" in node:
            for child in node["children"]:
                traverse(child)
    traverse(tree)
    return nodes

def node_name2titles(data: dict) -> dict:
    """
    Given a node dictionary with an aspect and its mapped segments,
    return a dictionary with the aspect_name and the list of selected segments.
    """
    node_name = data.get("label", "")
    indices = data.get("paper_ids", [])
#     collected_indices = []
#     perspectives = data.get("perspectives", {})
#     for key, value in perspectives.items():
#         if isinstance(value, dict) and "perspective_segments" in value:
#             segments = value.get("perspective_segments", [])
#             if isinstance(segments, list):
#                 collected_indices.extend(segments)
#     unique_indices = sorted(set(collected_indices))
#     if len(unique_indices) == len(mapped_segs):
#         selected_segments = mapped_segs
#     else:
#         print(f"In aspect {aspect_name}, only {len(unique_indices)} out of {len(mapped_segs)} segments are selected.")
#         selected_segments = [mapped_segs[i] for i in unique_indices if 0 <= i < len(mapped_segs)]
    return {"label": node_name, "indices": indices}
