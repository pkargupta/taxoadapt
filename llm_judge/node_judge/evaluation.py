# evaluation.py

from llm_judge.llm.io import llm_chat  # make sure this import path is correct in your project

def get_path_relevance(claim: str, paths: list) -> list:
    def get_prompt(claim, path):
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', decide whether this path from the aspect tree is relevant to the analysis of the claim: '{path}'\n\n"
            "Output options: '<relevant>' or '<irerelevant>'. Do some simple rationalization before giving the output if possible."
        )

    input_strs = [get_prompt(claim, path) for path in paths]
    outputs = llm_chat(input_strs, model_name="gpt-4o")
    
    results = []
    for path, output in zip(paths, outputs):
        result = {"path": path}
        if "<relevant>" in output.lower():
            result['score'] = 1
        elif "<irrelevant>" in output.lower():
            result['score'] = 0
        else:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    
    return results

def get_path_granularity(claim: str, paths: list) -> list:
    def get_prompt(claim, path):
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', decide whether this path from the aspect tree has good granularity: '{path}' Check whether the child node is a more specific subaspect of the parent node. \n\n"
            "Output options: '<good granularity>' or '<bad granularity>'. Do some simple rationalization before giving the output if possible."
        )
    input_strs = [get_prompt(claim, path) for path in paths]
    outputs = llm_chat(input_strs, model_name="gpt-4o")
    
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

def get_level_granularity(claim: str, levels: list) -> list:
    def get_prompt(claim, level_instance):
        parent = level_instance['parent']
        siblings = level_instance['siblings']
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', decide whether these siblings from parent node '{parent}', have good granularity: '{', '.join(siblings)}' Check whether they have similar specificity level. \n\n"
            "Output options: '<all not granular>' or '<majority not granular>' or "
            "'<majority granular>' or '<all granular>'. Do some simple rationalization before giving the output if possible."
        )
    input_strs = [get_prompt(claim, level) for level in levels]
    outputs = llm_chat(input_strs, model_name="gpt-4o")
    
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

def get_taxonomy_wise_uniqueness(claim: str, taxonomy: str) -> dict:
    def get_prompt(claim, taxonomy):
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', decide whether this taxonomy: {taxonomy} has overlapping nodes. \n\n"
            "Output options: '<overlapping>' or '<not overlapping>'. Do some simple rationalization before giving the output if possible."
        )
    prompt = get_prompt(claim, taxonomy)
    outputs = llm_chat([prompt], model_name="gpt-4o")[0]
    
    result = {"taxonomy": taxonomy}
    if "<overlapping>" in outputs.lower():
        result['score'] = 0
    elif "<not overlapping>" in outputs.lower():
        result['score'] = 1
    else:
        result['score'] = -1
    result['reasoning'] = outputs
    return result

def get_node_wise_segment_quality(claim: str, nodes: list) -> list:
    def get_prompt(claim, node):
        aspect_name = node['aspect_name']
        segments = node['segments']
        index_segments = "\n".join([f"{i+1}. {seg}" for i, seg in enumerate(segments)])
        return (
            "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
            "Particularly in scientific and political contexts. Instead, a claim can be broken down "
            "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
            f"Given the claim: '{claim}', Here is one aspect to analyze this claim: {aspect_name} and these are some segments: {index_segments}\n\n"
            "Count how many of them are relevant to this specific aspect.\n\n"
            "Output options: '<rel_seg_num> ... (int) </rel_seg_num>'. Do some rationalization before outputting the number of relevant segments."
        )
    
    input_strs = [get_prompt(claim, node) for node in nodes]
    outputs = llm_chat(input_strs, model_name="gpt-4o")
    results = []
    for node, output in zip(nodes, outputs):
        if len(node['segments']) == 0:
            continue
        result = {"node": node, "segments": node['segments']}
        try:
            output_int = output.split('<rel_seg_num>')[1].split('</rel_seg_num>')[0]
            result['score'] = int(output_int) / len(node['segments'])
        except Exception:
            result['score'] = -1
        result['reasoning'] = output
        results.append(result)
    return results
