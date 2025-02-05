import json
import argparse
import logging
from eval.llm.io import llm_chat

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_claim(json_path):
    """Load the claim from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get('aspect_name', None)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading claim from {json_path}: {e}")
        return None

def build_prompt(claim, height, node_num_per_level):
    """Construct the LLM prompt based on the claim and taxonomy height."""
    return (
        "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'—"
        "particularly in scientific and political contexts. Instead, a claim can be broken down "
        "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
        f"Given the claim: '{claim}', generate a taxonomy of the claim with a specified height of {height}.\n\n"
        f'Generate up to {node_num_per_level} subnodes per node in the taxonomy.\n'
        "The taxonomy should be structured as a dictionary, formatted as follows:\n"
        "{"
        '   "aspect_name": "the claim itself",  # the root aspect should be the claim itself (a sentence), but other aspects should be the aspect name (words or phrases)\n'
        '   "children": [\n'
        '       { "aspect_name": "Sub-aspect 1", "children": [...] },\n'
        '       { "aspect_name": "Sub-aspect 2", "children": [...] }, ...\n'
        "   ]\n"
        "}\n\n"
        "Ensure that the output is a valid JSON object, directly serializable using `json.loads()`. "
        "Do not include any extra formatting such as ```json."
    )

def generate_taxonomy(prompt, model_name):
    """Send the prompt to LLM and get taxonomy response."""
    response = llm_chat([prompt], model_name)
    try:
        return json.loads(response[0])
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON response: {e}")
        return None

def clean_taxonomy(node):
    """Recursively remove empty 'children' keys from the taxonomy."""
    if not isinstance(node, dict):
        return node
    node['children'] = [clean_taxonomy(child) for child in node.get('children', []) if child]
    if not node['children']:
        node.pop('children', None)
    return node

def save_taxonomy(output, output_path):
    """Save the cleaned taxonomy to a file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
        logging.info(f"Taxonomy saved successfully at {output_path}")
    except IOError as e:
        logging.error(f"Error saving taxonomy to {output_path}: {e}")

def main():
    """Main function to handle command-line execution."""
    parser = argparse.ArgumentParser(description="Generate a taxonomy from a claim using GPT model.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the LLM model")
    parser.add_argument("--height", type=int, default=2, help="Height of the taxonomy tree")
    parser.add_argument("--output_path", type=str, default="eval/example/zeroshot_taxonomy.json", help="Output file path")
    parser.add_argument("--input_path", type=str, default="eval/example/hierarchy.json", help="Input JSON file with claim")
    parser.add_argument("--node_num_per_level", type=int, default=5, help="Number of nodes per level in the taxonomy")

    args = parser.parse_args()

    # Load claim
    claim = load_claim(args.input_path)
    if not claim:
        logging.error("No valid claim found. Exiting.")
        return

    # Generate prompt
    prompt = build_prompt(claim, args.height, args.node_num_per_level)

    # Generate taxonomy from LLM
    taxonomy = generate_taxonomy(prompt, args.model_name)
    if not taxonomy:
        logging.error("Failed to generate a valid taxonomy. Exiting.")
        return

    # Clean taxonomy structure
    cleaned_taxonomy = clean_taxonomy(taxonomy)

    # Save the final taxonomy
    save_taxonomy(cleaned_taxonomy, args.output_path)

if __name__ == "__main__":
    main()
