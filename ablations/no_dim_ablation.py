import os
os.environ["VLLM_ATTENTION_BACKEND"] = "XFORMERS"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from collections import deque
from contextlib import redirect_stdout
import argparse
from tqdm import tqdm

from ..model_definitions import initializeLLM, promptLLM, constructPrompt
from ..prompts import multi_dim_prompt, NodeListSchema, type_cls_system_instruction, type_cls_main_prompt, TypeClsSchema
from ..taxonomy import Node, DAG
from datasets import load_dataset
from ..expansion import expandNodeWidth, expandNodeDepth
from ..paper import Paper
from ..utils import clean_json_string

def construct_dataset(args):
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    split = 'train'
    
    if args.dataset == 'emnlp_2024':
        ds = load_dataset("EMNLP/EMNLP2024-papers")
    elif args.dataset == 'emnlp_2022':
        ds = load_dataset("TimSchopf/nlp_taxonomy_data")
        split = 'test'
    elif args.dataset == 'cvpr_2024':
        ds = load_dataset("DeepNLP/CVPR-2024-Accepted-Papers")
    elif args.dataset == 'cvpr_2020':
        ds = load_dataset("DeepNLP/CVPR-2020-Accepted-Papers")
    elif args.dataset == 'iclr_2024':
        ds = load_dataset("DeepNLP/ICLR-2024-Accepted-Papers")
    elif args.dataset == 'iclr_2021':
        ds = load_dataset("DeepNLP/ICLR-2021-Accepted-Papers")
    elif args.dataset == 'icra_2024':
        ds = load_dataset("DeepNLP/ICRA-2024-Accepted-Papers")
    else:
        ds = load_dataset("DeepNLP/ICRA-2020-Accepted-Papers")
    
    
    internal_collection = {}

    with open(os.path.join(args.data_dir, 'internal.txt'), 'w') as i:
        internal_count = 0
        id = 0
        for p in tqdm(ds[split]):
            if ('title' not in p) and ('abstract' not in p):
                continue
            
            temp_dict = {"Title": p['title'], "Abstract": p['abstract']}
            formatted_dict = json.dumps(temp_dict)
            i.write(f'{formatted_dict}\n')
            internal_collection[id] = Paper(id, p['title'], p['abstract'], label_opts=args.dimensions, internal=True)
            internal_count += 1
            id += 1
        print("Total # of Papers: ", internal_count)
    
    return internal_collection, internal_count

def initialize_DAG(args):
    ## we want to make this a directed acyclic graph (DAG) so maintain a list of the nodes
    roots = {}
    id2node = {}
    label2node = {}
    idx = 0

    for dim in args.dimensions:
        mod_topic = args.topic.replace(' ', '_').lower()
        mod_full_topic = args.topic.replace(' ', '_').lower() + f"_{dim}"
        root = Node(
                id=idx,
                label=mod_topic,
                dimension=dim
            )
        roots[dim] = root
        id2node[idx] = root
        label2node[mod_full_topic] = root
        idx += 1

    queue = deque([node for id, node in id2node.items()])

    # if taking long, you can probably parallelize this between the different taxonomies (expand by level)
    while queue:
        curr_node = queue.popleft()
        label = curr_node.label
        dim = curr_node.dimension
        # expand
        system_instruction, main_prompt, json_output_format = multi_dim_prompt(curr_node)
        prompts = [constructPrompt(args, system_instruction, main_prompt + "\n\n" + json_output_format)]
        outputs = promptLLM(args=args, prompts=prompts, schema=NodeListSchema, max_new_tokens=3000, json_mode=True, temperature=0.01, top_p=1.0)[0]
        outputs = json.loads(clean_json_string(outputs)) if "```" in outputs else json.loads(outputs.strip())
        outputs = outputs['root_topic'] if 'root_topic' in outputs else outputs[label]

        # add all children
        for key, value in outputs.items():
            mod_key = key.replace(' ', '_').lower()
            mod_full_key = mod_key + f"_{dim}"
            if mod_full_key not in label2node:
                child_node = Node(
                        id=len(id2node),
                        label=mod_key,
                        dimension=dim,
                        description=value['description'],
                        parents=[curr_node]
                    )
                curr_node.add_child(mod_key, child_node)
                id2node[child_node.id] = child_node
                label2node[mod_full_key] = child_node
                if child_node.level < args.init_levels:
                    queue.append(child_node)
            elif label2node[mod_full_key] in label2node[label + f"_{dim}"].get_ancestors():
                continue
            else:
                child_node = label2node[mod_full_key]
                curr_node.add_child(mod_key, child_node)
                child_node.add_parent(curr_node)

    return roots, id2node, label2node


def main(args):

    print("######## STEP 1: LOAD IN DATASET ########")

    internal_collection, internal_count = construct_dataset(args)
    
    print(f'Internal: {internal_count}')

    print("######## STEP 2: INITIALIZE DAG ########")
    args = initializeLLM(args)

    roots, id2node, label2node = initialize_DAG(args)

    for dim in args.dimensions:
        with open(f'{args.data_dir}/initial_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                roots[dim].display(0, indent_multiplier=5)

    print("######## STEP 3: DO NOT PARTITION THE PAPERS BY DIMENSION!")

    args.llm = 'vllm'
    dags = {dim:DAG(root=root, dim=dim) for dim, root in roots.items()}

    # do for internal collection
    for r in roots:
        roots[r].papers = {}
    type_dist = {dim:[] for dim in args.dimensions}

    for p_id in range(len(internal_collection)):
        internal_collection[p_id].labels = {}
        for dim in args.dimensions:
            type_dist[dim].append(internal_collection[p_id])
            internal_collection[p_id].labels[dim] = []
            roots[dim].papers[p_id] = internal_collection[p_id]
    
    print(str({k:len(v) for k,v in type_dist.items()}))


    # for each node, classify its papers for the children or perform depth expansion
    print("######## STEP 4: ITERATIVELY CLASSIFY & EXPAND ########")

    visited = set()
    queue = deque([roots[r] for r in roots])

    while queue:
        curr_node = queue.popleft()
        print(f'VISITING {curr_node.label} ({curr_node.dimension}) AT LEVEL {curr_node.level}. WE HAVE {len(queue)} NODES LEFT IN THE QUEUE!')
        
        if len(curr_node.children) > 0:
            if curr_node.id in visited:
                continue
            visited.add(curr_node.id)

            # classify
            curr_node.classify_node(args, label2node, visited)

            # sibling expansion if needed
            new_sibs = expandNodeWidth(args, curr_node, id2node, label2node)
            print(f'(WIDTH EXPANSION) new children for {curr_node.label} ({curr_node.dimension}) are: {str((new_sibs))}')

            # re-classify and re-do process if necessary
            if len(new_sibs) > 0:
                curr_node.classify_node(args, label2node, visited)
            
            # add children to queue if constraints are met
            for child_label, child_node in curr_node.children.items():
                c_papers = label2node[child_label + f"_{curr_node.dimension}"].papers
                if (child_node.level < args.max_depth) and (len(c_papers) > args.max_density):
                    queue.append(child_node)
        else:
            # no children -> perform depth expansion
            new_children, success = expandNodeDepth(args, curr_node, id2node, label2node)
            args.llm = 'vllm'
            print(f'(DEPTH EXPANSION) new {len(new_children)} children for {curr_node.label} ({curr_node.dimension}) are: {str((new_children))}')
            if (len(new_children) > 0) and success:
                queue.append(curr_node)
    
    print("######## STEP 5: SAVE THE TAXONOMY ########")
    for dim in args.dimensions:
        with open(f'{args.data_dir}/final_taxo_{dim}.txt', 'w') as f:
            with redirect_stdout(f):
                taxo_dict = roots[dim].display(0, indent_multiplier=5)

        with open(f'{args.data_dir}/final_taxo_{dim}.json', 'w', encoding='utf-8') as f:
            json.dump(taxo_dict, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', type=str, default='natural language processing')
    parser.add_argument('--dataset', type=str, default='llm_graph')
    parser.add_argument('--llm', type=str, default='gpt')
    parser.add_argument('--max_depth', type=int, default=2)
    parser.add_argument('--init_levels', type=int, default=1)
    parser.add_argument('--max_density', type=int, default=40)

    args = parser.parse_args()

    args.dimensions = ["tasks", "datasets", "methodologies", "evaluation_methods", "real_world_domains"]

    args.dataset = "emnlp_2022"
    args.topic = "natural language processing"
    args.data_dir = f"datasets/{args.dataset.lower().replace(' ', '_')}_nodim"
    args.internal = f"{args.dataset}.txt"

    main(args)