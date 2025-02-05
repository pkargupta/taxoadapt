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


def build_prompt(claim, literature, max_aspects_per_node=5):
    """
    Construct the LLM prompt based on the claim and taxonomy height.
    
    Args:
        claim (str): The current claim.
        literature (str): The top relevant literature segments.
        max_aspects_per_node (int): The maximum number of aspects to generate.

    Returns:
        str: The prompt to be passed to the language model.
    """
    return (
        "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized "
        "as entirely 'true' or 'false'—particularly in scientific and political contexts. Instead, a claim can "
        "be broken down into its core aspects, which are easier to evaluate individually.\n\n"
        f"Given the claim: '{claim}', generate a list of up to {max_aspects_per_node} aspects of the claim. "
        "They should be the node of the same level\n\n"
        f"Here are some literautre segments to help you generate the aspects: {literature}\n"
        "The aspects should be structured as a list, formatted as follows:\n"
        "['keyword 1', 'keyword 2', 'keyword 3']\n"
        "Ensure that the output is a valid JSON object, directly serializable using json.loads(). "
        "Do not include any extra formatting such as ```json."
    )


def clean_taxonomy(node):
    """
    Recursively remove empty 'children' keys from the taxonomy.
    
    Args:
        node (dict): A node in the taxonomy tree.

    Returns:
        dict: A cleaned version of the node with no empty children lists.
    """
    if not isinstance(node, dict):
        return node
    node['children'] = [clean_taxonomy(child) for child in node.get('children', []) if child]
    if not node['children']:
        node.pop('children', None)
    return node


def save_taxonomy(output, output_path):
    """
    Save the cleaned taxonomy to a file.
    
    Args:
        output (dict): The taxonomy data to be saved.
        output_path (str): The path where the taxonomy will be saved.
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


def iterative_rag(args, current_claim, original_claim, segment_embeddings, embedding_func, current_height=0):
    """
    Recursively build a taxonomy for a claim by splitting it into aspects at each level,
    guided by the most relevant literature segments.

    Args:
        args: Parsed command-line arguments.
        current_claim (str): The current claim or sub-claim being processed.
        original_claim (str): The main/initial claim.
        segment_embeddings (dict): Precomputed embeddings for literature segments.
        embedding_func (callable): Function to produce embeddings for a given text.
        current_height (int): Current depth in the taxonomy tree.

    Returns:
        dict: A dictionary representing the node at this level and its children.
    """
    print("Current claim: ", current_claim)
    print("Current height: ", current_height)

    # Determine the aspect name for the node
    if current_claim == original_claim:
        aspect_name = current_claim
    else:
        aspect_name = current_claim.split("With regard to ")[1].split(", ")[0]

    # Base case: if we have reached the maximum height
    if current_height == args.height:
        return {"aspect_name": aspect_name}

    # Embed the current claim
    claim_embedding = embedding_func([current_claim])[current_claim]

    # Find the most similar segments to the claim
    segment_similarities = []
    for segment, segment_embedding in segment_embeddings.items():
        similarity = cosine_similarity(
            claim_embedding.reshape(1, -1),
            segment_embedding.reshape(1, -1)
        )
        segment_similarities.append((segment, similarity))

    # Sort by descending similarity and select top segments
    segment_similarities = sorted(segment_similarities, key=lambda x: x[1], reverse=True)
    top_segments = [seg for seg, _ in segment_similarities[:args.rag_segment_num]]
    literature = '\n'.join(top_segments)

    # Build the prompt and call the LLM
    input_prompt = build_prompt(current_claim, literature, args.max_aspects_per_node)
    response = llm_chat([input_prompt], model_name=args.model_name)[0]

    # Attempt to parse the LLM response as JSON
    try:
        if '```json' in response:
            response = response.strip()[7:-3]
        aspects = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON response: {e}")
        raise e

    # Recursively process each aspect
    children = [
        iterative_rag(
            args,
            f"With regard to {aspect}, {original_claim}",
            original_claim,
            segment_embeddings,
            embedding_func,
            current_height + 1
        )
        for aspect in aspects
    ]

    return {"aspect_name": aspect_name, "children": children}


def main():
    """
    Main function to handle command-line execution:
      1. Parse arguments.
      2. Load the claim.
      3. Load or create and cache embeddings for literature segments.
      4. Generate a taxonomy from the claim using a recursive approach.
      5. Clean and save the final taxonomy.
    """
    parser = argparse.ArgumentParser(description="Generate a taxonomy from a claim using GPT model.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the LLM model")
    parser.add_argument("--height", type=int, default=2, help="Height of the taxonomy tree")
    parser.add_argument("--output_path", type=str, default="eval/example/rag_iterative_taxonomy.json", help="Output file path")
    parser.add_argument("--input_path", type=str, default="eval/example/hierarchy.json", help="Input JSON file with claim")
    parser.add_argument("--segment_embed_cache_path", type=str, default=".cache/vaccine_seg_embed_cache.pickle", help="Cache directory for embedding")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--topic", default="vaccine")
    parser.add_argument("--rag_segment_num", default=20)
    parser.add_argument("--max_aspects_per_node", default=5)
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

    # Generate the taxonomy
    taxonomy = iterative_rag(args, claim, claim, segment_embeddings, embedding_func)
    if not taxonomy:
        logging.error("Failed to generate a valid taxonomy. Exiting.")
        return

    # Clean and save the taxonomy
    cleaned_taxonomy = clean_taxonomy(taxonomy)
    save_taxonomy(cleaned_taxonomy, args.output_path)


if __name__ == "__main__":
    main()
