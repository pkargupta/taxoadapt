import argparse
import json
import logging
import os
import pickle

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

from api.local.e5_model import E5
from eval.llm.io import llm_chat

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def get_embedding_model(model_name: str = 'e5'):
    """
    Return an embedding function for the specified model name.
    Currently only supports the 'e5' model.
    """
    if model_name == 'e5':
        e5 = E5()

        def embed(text: list[str]):
            return e5(text)

        return embed
    else:
        raise ValueError(f"Model {model_name} not supported.")


def load_claim(json_path):
    """
    Load the claim from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing the claim.

    Returns:
        str or None: The 'aspect_name' value if present, otherwise None.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('aspect_name', None)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading claim from {json_path}: {e}")
        return None


def build_prompt(claim, height, literature: str, node_num_per_level):
    """
    Construct the LLM prompt based on the claim and taxonomy height.

    Args:
        claim (str): The main claim to be analyzed.
        height (int): The height of the desired taxonomy tree.
        literature (str): Relevant literature segments.

    Returns:
        str: The prompt to be passed to the language model.
    """
    return (
        "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized "
        "as entirely 'true' or 'false'—particularly in scientific and political contexts. Instead, a claim "
        "can be broken down into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
        f"Given the claim: '{claim}', generate a taxonomy of the claim with a specified height of {height}.\n\n"
        f"Here are some literautre segments to help you generate the taxonomy: {literature}\n"
        f'Generate up to {node_num_per_level} subnodes per node in the taxonomy.\n'
        "The taxonomy should be structured as a dictionary, formatted as follows:\n"
        "{"
        '   "aspect_name": "the claim itself",  # the root aspect should be the claim itself (a sentence), '
        'but other aspects should be the aspect name (words or phrases)\n'
        '   "children": [\n'
        '       { "aspect_name": "Sub-aspect 1", "children": [...] },\n'
        '       { "aspect_name": "Sub-aspect 2", "children": [...] }, ...\n'
        "   ]\n"
        "}\n\n"
        "Ensure that the output is a valid JSON object, directly serializable using json.loads(). "
        "Do not include any extra formatting such as ```json."
    )


def generate_taxonomy(prompt, model_name):
    """
    Send the prompt to the LLM and return the parsed taxonomy response.

    Args:
        prompt (str): Prompt to be sent to the language model.
        model_name (str): Name of the LLM model.

    Returns:
        dict or None: Parsed JSON if successful, otherwise None.
    """
    response = llm_chat([prompt], model_name)
    try:
        return json.loads(response[0])
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        return None


def clean_taxonomy(node):
    """
    Recursively remove empty 'children' keys from the taxonomy.

    Args:
        node (dict): A node in the taxonomy tree.

    Returns:
        dict: A cleaned version of the node, without empty children lists.
    """
    if not isinstance(node, dict):
        return node
    node['children'] = [clean_taxonomy(child) for child in node.get('children', []) if child]
    if not node['children']:
        node.pop('children', None)
    return node


def save_taxonomy(output, output_path):
    """
    Save the cleaned taxonomy to a JSON file.

    Args:
        output (dict): The taxonomy data to be saved.
        output_path (str): File path for saving the output JSON.
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
        logging.info(f"Taxonomy saved successfully at {output_path}")
    except IOError as e:
        logging.error(f"Error saving taxonomy to {output_path}: {e}")


def load_segments(args, chunk_size=3):
    """
    Load text segments from a topic-specific text file, chunked by a specified size.

    Args:
        args: Parsed command-line arguments.
        chunk_size (int): Number of sentences to group into one segment.

    Returns:
        list: List of segmented and cleaned text chunks.
    """
    corpus = []
    file_path = f"{args.data_dir}/{args.topic}/{args.topic}_text.txt"

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            sents = line.strip().lower().split('. ')
            for i in np.arange(0, len(sents), chunk_size):
                seg_content = unidecode(". ".join(sents[i:i + chunk_size]))
                corpus.append(seg_content)
    return corpus


def main():
    """
    Main function to handle command-line execution:
      1. Parse arguments.
      2. Load the claim.
      3. Load or compute embeddings for literature segments.
      4. Select the most relevant segments.
      5. Build and send a prompt to the LLM.
      6. Clean and save the resulting taxonomy.
    """
    parser = argparse.ArgumentParser(description="Generate a taxonomy from a claim using GPT model.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the LLM model")
    parser.add_argument("--height", type=int, default=2, help="Height of the taxonomy tree")
    parser.add_argument("--output_path", type=str, default="eval/example/rag_base_taxonomy.json", help="Output file path")
    parser.add_argument("--input_path", type=str, default="eval/example/hierarchy.json", help="Input JSON file with claim")
    parser.add_argument("--segment_embed_cache_path", type=str, default=".cache/vaccine_seg_embed_cache.pickle", help="Cache directory for embedding")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--topic", default="vaccine")
    parser.add_argument("--rag_segment_num", default=20)
    parser.add_argument("--node_num_per_level", default=5)
    args = parser.parse_args()

    # Load the claim
    claim = load_claim(args.input_path)
    if not claim:
        logging.error("No valid claim found. Exiting.")
        return

    # Load text segments
    segments = load_segments(args)

    # Define an embedding function
    embedding_func = get_embedding_model()

    # Load or compute segment embeddings
    if os.path.exists(args.segment_embed_cache_path):
        with open(args.segment_embed_cache_path, 'rb') as f:
            segment_embeddings = pickle.load(f)
    else:
        segment_embeddings = embedding_func(segments)
        with open(args.segment_embed_cache_path, 'wb') as f:
            pickle.dump(segment_embeddings, f)

    # Embed the main claim
    claim_embedding = embedding_func([claim])[claim]

    # Find and select the most relevant segments
    segment_similarity = []
    for segment_str, segment_embed in segment_embeddings.items():
        similarity = cosine_similarity([claim_embedding], [segment_embed])[0][0]
        segment_similarity.append((segment_str, similarity))

    segment_similarity = sorted(segment_similarity, key=lambda x: x[1], reverse=True)
    selected_segments = [seg for seg, sim in segment_similarity[:args.rag_segment_num]]
    literature = "\n".join(selected_segments)

    # Build the prompt and generate the taxonomy
    prompt = build_prompt(claim, args.height, literature, args.node_num_per_level)
    taxonomy = generate_taxonomy(prompt, args.model_name)
    if not taxonomy:
        logging.error("Failed to generate a valid taxonomy. Exiting.")
        return

    # Clean and save the taxonomy
    cleaned_taxonomy = clean_taxonomy(taxonomy)
    save_taxonomy(cleaned_taxonomy, args.output_path)


if __name__ == "__main__":
    main()
