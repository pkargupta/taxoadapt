# evaluation.py

# from gpt import gpt4o_chat as llm_chat  # make sure this import path is correct in your project
from api.openai.chat import chat
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def get_dimension_alignment(root: str, dim: str, nodes: list) -> list:
    def get_prompt(claim, node):
        return (
            "Scientific concepts are naturally organized in multi-dimensional taxonomic structures, with more specific concepts being the children of a broader research topic.\n\n"
            f"Given the root topic: '{root}', decide whether this node from a taxonomy is relevant to the {dim} aspect of the root topic: '{node}'\n\n"
            "Output options: '<relevant>' or '<irerelevant>'. Do some simple rationalization before giving the output if possible."
        )

    input_strs = [get_prompt(root, node['label']) for node in nodes]
    outputs = chat(input_strs, model_name="gpt-4o", seed=42)
    
    results = []
    for node, output in zip(nodes, outputs):
        result = {"node": node['label']}
        if "<relevant>" in output.lower():
            result['score'] = 1
        elif "<irrelevant>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

def get_path_granularity(root: str, paths: list) -> list:
    def get_prompt(claim, path):
        return (
            "Scientific concepts are naturally organized in multi-dimensional taxonomic structures, with more specific concepts being the children of a broader research topic.\n\n"
            f"Given the root topic: '{root}', decide whether this path from the scientific concept taxonomy has good granularity: '{path}' Check whether the child node is a more specific subaspect of the parent node. \n\n"
            "Output options: '<good granularity>' or '<bad granularity>'. Do some simple rationalization before giving the output if possible."
        )
    input_strs = [get_prompt(root, path) for path in paths]
    outputs = chat(input_strs, model_name="gpt-4o", seed=42)
    
    results = []
    for path, output in zip(paths, outputs):
        result = {"path": path}
        if "<good granularity>" in output.lower():
            result['score'] = 1
        elif "<bad granularity>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

def get_level_granularity(root: str, levels: list) -> list:
    def get_prompt(root, level_instance):
        parent = level_instance['parent']
        siblings = level_instance['siblings']
        return (
            "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
            f"Given the root topic: '{root}', decide whether these siblings from parent node '{parent}', have good granularity: '{', '.join(siblings)}' Check whether they have similar specificity level. \n\n"
            "Output options: '<all not granular>' or '<majority not granular>' or "
            "'<majority granular>' or '<all granular>'. Do some simple rationalization before giving the output if possible."
        )
    input_strs = [get_prompt(root, level) for level in levels]
    outputs = chat(input_strs, model_name="gpt-4o", seed=42)
    
    results = []
    for level_instance, output in zip(levels, outputs):
        result = {"path": level_instance}
        if "<all not granular>" in output.lower():
            result['score'] = 1
        elif "<majority not granular>" in output.lower():
            result['score'] = 2
        elif "<majority granular>" in output.lower():
            result['score'] = 3
        elif "<all granular>" in output.lower():
            result['score'] = 4
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

def get_level_granularity_new(root: str, levels: list) -> list:
    def get_prompt(root, level_instance):
        parent = level_instance['parent']
        siblings = level_instance['siblings']
        return (
            f"You are determining the coherence of a set of {root} subtopics of the parent topic {parent}.\n\nThe parent topic is: {parent}.\n\nThe set of siblings, which are child subtopics of the parent, are: '{', '.join(siblings)}'\n\nEvaluate the overall coherence of the sibling set based on their collective specificity and granularity relative to {parent}. Use the following scoring criteria:\n\n"
            f"Score=<no_sibling_coherence>: The set is highly inconsistent or incoherent (only one subtopic), with most topics significantly misaligned in specificity relative to the parent.\n"
            f"Score=<weak_sibling_coherence>: The set shows considerable inconsistency, with several topics deviating noticeably from the expected level of specificity.\n"
            f"Score=<reasonable_sibling_coherence>: The set is generally coherent, with only minor inconsistencies in specificity among the topics.\n"
            f"Score=<strong_sibling_coherence>: The set is fully coherent, with all topics properly matching the expected level of specificity and granularity for the parent.\n"
            "Output options: '<no_sibling_coherence>', '<weak_sibling_coherence>', '<reasonable_sibling_coherence>', or '<strong_sibling_coherence>'. Do some simple rationalization before giving the output if possible."
        )

    input_strs = [get_prompt(root, level) for level in levels]
    outputs = chat(input_strs, model_name="gpt-4o", seed=42)
    
    results = []
    for level_instance, output in zip(levels, outputs):
        result = {"path": level_instance}
        if "<no_sibling_coherence>" in output.lower():
            result['score'] = 0
        elif "<weak_sibling_coherence>" in output.lower():
            result['score'] = 1/3
        elif "<reasonable_sibling_coherence>" in output.lower():
            result['score'] = 2/3
        elif "<strong_sibling_coherence>" in output.lower():
            result['score'] = 1
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

# def get_node_wise_uniqueness(root: str, nodes: list, taxonomy: str) -> dict:
#     def get_prompt(root, node_name, taxonomy):
#         return (
#             "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
#             f"Given the root topic: '{root}', decide whether this node: {node_name} has other overlapping nodes in the taxonomy which cover the same research topic: {taxonomy} \n\n"
#             "A subtopic and its parent are not considered as overlapping."
#             "Output options: '<overlapping>' or '<not overlapping>'. Do some simple rationalization before giving the output if possible."
#         )
#     prompts = [get_prompt(root, node['label'], taxonomy) for node in nodes]
#     outputs = llm_chat(prompts, model_name="gpt-4o")
    
# #     result = {"taxonomy": taxonomy}
#     results = []
#     for node, output in zip(nodes, outputs):
#         result = {'node': node['label']}
#         if "<overlapping>" in output.lower():
#             result['score'] = 0
#         elif "<not overlapping>" in output.lower():
#             result['score'] = 1
#         else:
#             result['score'] = -1
#         result['reasoning'] = output
#         results.append(result)
#     return results

def get_node_wise_uniqueness_equivalent(root: str, nodes: list, taxonomy: str) -> dict:
    def get_prompt(root, node_name, taxonomy):
        return (
            "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
            f"Given the root topic: '{root}', decide whether this node: {node_name} has other equivalent nodes in the taxonomy which cover the same research topic: {taxonomy} \n\n"
            "A subtopic and its parent are not considered as equivalent."
            "Output options: '<has equivalent>' or '<no equivalent>'. Do some simple rationalization before giving the output if possible."
        )
    prompts = [get_prompt(root, node['label'], taxonomy) for node in nodes]
    outputs = chat(prompts, model_name="gpt-4o", seed=42)
    
#     result = {"taxonomy": taxonomy}
    results = []
    for node, output in zip(nodes, outputs):
        result = {'node': node['label']}
        if "<has equivalent>" in output.lower():
            result['score'] = 0
        elif "<no equivalent>" in output.lower():
            result['score'] = 1
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    return results

def get_node_wise_segment_quality(root: str, nodes: list, id2paper: dict) -> list:
    def get_prompt(root, node, indices, batch_size=100):
        node_name = node['label']
#         indices = node['indices']
        prompts = []
        batch_idx = []
        for batch in range((len(indices)-1) // batch_size + 1):
            papers = [(i, id2paper[i]['Title'], id2paper[i]['Abstract']) for i in indices[batch*batch_size:(batch+1)*batch_size] if i in id2paper]
            index_papers = "\n".join([f"{i}. Title: {title}. Abstract: {abstract}" for (i, title, abstract) in papers])
#             index_papers = "\n".join([f"{i+1}. {title}" for i, (title, abstract) in enumerate(papers)])
#             prompts.append(
#                 "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
#                 f"Given the root topic: '{root}', here is one of its subtopics: {node_name} and these are some papers: {index_papers}\n\n"
#                 "Count how many of them are relevant to this specific subtopic.\n\n"
#                 "Output options: '<rel_paper_num> ... (int) </rel_paper_num>'. Do some rationalization before outputting the number of relevant segments."
#             )
            prompts.append(
                "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
                f"Given the root topic: '{root}', here is one of its subtopics: {node_name} and these are some papers: {index_papers}\n\n"
                "Provide a list of paper IDs that are relevant to this subtopic.\n\n"
                "Output options: '<rel_paper> ID1, ID2, ... </rel_paper>'. Do some rationalization before outputting the list of relevant paper IDs."
            )
            batch_idx.append([i for i in indices[batch*batch_size:(batch+1)*batch_size] if i in id2paper])
        return prompts, batch_idx
    
    no_root = [node for node in nodes if node['label'] != root]
    
    results = {}
    for node in tqdm(no_root):
        if len(node['indices']) == 0:
            continue
        results[node['label']] = {"node": node['label'], 'relevant':[], 'valid':[], 'reasoning':[]}
        prompts, batches = get_prompt(root, node, node['indices'], 100)
        outputs = llm_chat(prompts, model_name="gpt-4o", verbose=0)
        for output, batch_idx in zip(outputs, batches):
            try:
                output_idx = output.split('<rel_paper>')[1].split('</rel_paper>')[0]
                sub_outputs = [(output, batch_idx)]
            except:
                sub_prompts, sub_batches = get_prompt(root, node, batch_idx, 10)
                sub_outputs = llm_chat(sub_prompts, model_name="gpt-4o", verbose=0)
                sub_outputs = [(o, b) for o, b in zip(sub_outputs, sub_batches)]
            for output, batch in sub_outputs:
                try:
                    output_idx = output.split('<rel_paper>')[1].split('</rel_paper>')[0]
                except:
                    continue
                results[node['label']]['valid'].extend(batch)
                for i in output_idx.strip().split(', '):
                    try:
                        results[node['label']]['relevant'].append(int(i))
                    except Exception:
                        continue
        results[node['label']]['reasoning'].append(sub_outputs)
    return list(results.values())
                
    
#     input_strs = [get_prompt(root, node, 10) for node in no_root]
#     flatten_nodes, flatten_strs = [], []
#     for node, prompts in zip(no_root, input_strs):
#         for prompt in prompts:
#             flatten_nodes.append(node)
#             flatten_strs.append(prompt)
#     outputs = llm_chat(flatten_strs, model_name="gpt-4o")
#     results = {}
#     for node, output in zip(flatten_nodes, outputs):
#         if len(node['indices']) == 0:
#             continue
#         if node['label'] not in results:
#             results[node['label']] = {"node": node, 'relevant':[], 'reasoning':[]}
#         try:
#             output_idx = output.split('<rel_paper>')[1].split('</rel_paper>')[0]
#         except:
#             continue
#         for i in output_idx.strip().split(', '):
#             try:
#                 results[node['label']]['relevant'].append(int(i))
#             except Exception:
#                 continue
#         results[node['label']]['reasoning'].append(output)
# #         results.append(result)
#     return list(results.values())



def get_node_wise_segment_quality_per_paper(root: str, nodes: list, id2paper: dict) -> list:
    def get_prompt(root, node):
        node_name = node['label']
        indices = node['indices']
        prompts = []
        for idx in indices:
            paper = (id2paper[idx]['Title'], id2paper[idx]['Abstract'])
            paper = f"Title: {paper[0]}\n Abstract: {paper[1]}"
            prompts.append(
                "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
                f"Given the root topic: '{root}' and one of its subtopics: {node_name}, decide whether this paper is relevant to this subtopic: {paper}\n\n"
                "Output options: '<relevant>' or '<irerelevant>'. Do some simple rationalization before giving the output if possible."
            )
        return prompts
    
    results = []
    for node in nodes:
        if len(node['indices']) == 0 or node['label'] == root:
            continue
        prompts = get_prompt(root, node)
        outputs = llm_chat(prompts, model_name="gpt-4o")
        result = {'node': node, 'scores':[]}
        for idx, output in zip(node['indices'], outputs):
            if "<relevant>" in output.lower():
                result['scores'].append(1)
            elif "<irrelevant>" in output.lower():
                result['scores'].append(0)
            else:
                result['scores'].append(-1)
        results.append(result)
    return results
            
    
def get_node_wise_paper_relevance(root: str, nodes: list, id2paper: dict, min_sup=10) -> list:
    def get_prompt(root, node, indices, batch_size=100):
        node_name = node['label']
#         indices = node['indices']
        prompts = []
        batch_idx = []
        for batch in range((len(indices)-1) // batch_size + 1):
            papers = [(i, id2paper[i]['Title'], id2paper[i]['Abstract']) for i in indices[batch*batch_size:(batch+1)*batch_size] if i in id2paper]
            index_papers = "\n".join([f"{i}. Title: {title}. Abstract: {abstract}" for (i, title, abstract) in papers])
            prompts.append(
                "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
                f"Given the root topic: '{root}', here is one of its subtopics: {node_name} and these are some papers: {index_papers}\n\n"
                "Provide a list of paper IDs that are relevant to this subtopic.\n\n"
                "Output options: '<rel_paper> ID1, ID2, ... </rel_paper>'. Do some rationalization before outputting the list of relevant paper IDs."
            )
            batch_idx.append([i for i in indices[batch*batch_size:(batch+1)*batch_size] if i in id2paper])
        return prompts, batch_idx
    
    no_root = [node for node in nodes if node['label'] != root]
    results = {}
    for node in tqdm(no_root):
        node_indices = node['indices'] + [i for i in id2paper.keys() if i not in set(node['indices'])]
        prompts, _ = get_prompt(root, node, node_indices, 100)
        results[node['label']] = {"node": node['label'], 'relevant':[], 'reasoning':[]}
        for prompt in (prompts):
            output = llm_chat([prompt], model_name="gpt-4o", verbose=0)[0]
            try:
                output_idx = output.split('<rel_paper>')[1].split('</rel_paper>')[0]
            except:
                print(output)
                continue
            for i in output_idx.strip().split(', '):
                try:
                    results[node['label']]['relevant'].append(int(i))
                except Exception:
                    continue
            results[node['label']]['reasoning'].append(output)
            if len(results[node['label']]['relevant']) >= min_sup:
                break
    return list(results.values())
            

def get_paper_coverage(root: str, indices: list, taxonomy: str, id2paper) -> dict:
    def get_prompt(root, paper, taxonomy):
        paper_content = "{}. {}".format(paper['Title'], paper['Abstract'])
        return (
            "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
            f"Given the root topic: '{root}', decide whether this paper: '{paper_content}' is relevant to at least one node in the taxonomy (exclude the root node): {taxonomy} \n\n"
            "Output options: '<relevant>' or '<not relevant>'. Do some simple rationalization before giving the output if possible."
        )
    prompts = [get_prompt(root, id2paper[i], taxonomy) for i in indices]
    outputs = chat(prompts, model_name="gpt-4o", seed=42)
    
    results = []
    for idx, output in zip(indices, outputs):
        result = {'paper_id': idx}
        if "<relevant>" in output.lower():
            result['score'] = 1
        elif "<not relevant>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    return results

def get_node_wise_paper_relevance_all(root: str, nodes: list, id2paper: dict) -> list:
    def get_prompt(root, node, indices, batch_size=100):
        node_name = node['label']
#         indices = node['indices']
        prompts = []
#         batch_idx = []
        for batch in range((len(indices)-1) // batch_size + 1):
            papers = [(i, id2paper[i]['Title'], id2paper[i]['Abstract']) for i in indices[batch*batch_size:(batch+1)*batch_size] if i in id2paper]
            index_papers = "\n".join([f"{i}. Title: {title}. Abstract: {abstract}" for (i, title, abstract) in papers])
            prompts.append(
                "Scientific concepts are naturally organized in a multi-dimensional taxonomic structure, with more specific concepts being the children of a broader research topic.\n\n"
                f"Given the root topic: '{root}', here is one of its subtopics: {node_name} and these are some papers: {index_papers}\n\n"
                "Provide a list of paper IDs that are relevant to this subtopic.\n\n"
                "Output options: '<rel_paper> ID1, ID2, ... </rel_paper>'. Do some rationalization before outputting the list of relevant paper IDs."
            )
#             batch_idx.append([i for i in indices[batch*batch_size:(batch+1)*batch_size] if i in id2paper])
        return prompts
    
    no_root = [node for node in nodes if node['label'] != root]
    
    
    input_strs = [get_prompt(root, node, list(id2paper.keys()), 100) for node in no_root]
    flatten_nodes, flatten_strs = [], []
    for node, prompts in zip(no_root, input_strs):
        for prompt in prompts:
            flatten_nodes.append(node)
            flatten_strs.append(prompt)
    outputs = chat(flatten_strs, model_name="gpt-4o-mini", seed=42)
    results = {}
    for node, output in zip(flatten_nodes, outputs):
        try:
            output_idx = output.split('<rel_paper>')[1].split('</rel_paper>')[0]
        except:
            continue
        if node['label'] not in results:
            results[node['label']] = {"node": node['label'], 'relevant':[], 'reasoning':[]}
        for i in output_idx.strip().split(', '):
            try:
                paper_id = int(i)
                if paper_id in id2paper:
                    results[node['label']]['relevant'].append(paper_id)
            except Exception:
                continue
        results[node['label']]['reasoning'].append(output)
    return list(results.values())
#     results = {}
#     for node in no_root:
#         node_indices = node['indices'] + [i for i in id2paper.keys() if i not in set(node['indices'])]
#         prompts, _ = get_prompt(root, node, node_indices, 100)
#         results[node['label']] = {"node": node['label'], 'relevant':[], 'reasoning':[]}
#         outputs = chat(prompts, model_name="gpt-4o", seed=42)
#         for output in outputs:
#             try:
#                 output_idx = output.split('<rel_paper>')[1].split('</rel_paper>')[0]
#             except:
#                 continue
#             for i in output_idx.strip().split(', '):
#                 try:
#                     paper_id = int(i)
#                     if paper_id in id2paper:
#                         results[node['label']]['relevant'].append(paper_id)
#                 except Exception:
#                     continue
#             results[node['label']]['reasoning'].append(output)
# #             if len(results[node['label']]['relevant']) >= min_sup:
# #                 break
#     return list(results.values())