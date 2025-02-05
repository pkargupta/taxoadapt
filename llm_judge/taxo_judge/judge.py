import json
import argparse
from enum import Enum
from dataclasses import dataclass, asdict
from llm_judge.llm.io import llm_chat


class SingleResult(Enum):
    BASELINE_WINS = "baseline wins"
    METHOD_WINS = "method wins"
    TIE = "tie"
    INVALID = "invalid"


@dataclass
class JudgeResults:
    baseline_name: str
    method_name: str
    rationale: list[str]
    result: SingleResult


def get_json_files(args):
    baseline_json_list = []
    for path in args.baseline_json_path_list:
        with open(path, 'r') as f:
            baseline_json_list.append(json.load(f))

    with open(args.method_json_path, 'r') as f:
        method_json = json.load(f)
    return baseline_json_list, method_json


def get_prompt(claim, taxo_str1, taxo_str2) -> str:
    """Construct the LLM prompt based on the claim and taxonomy height."""
    return (
        "Claims made by individuals or entities are often nuanced and cannot always be strictly categorized as entirely 'true' or 'false'. "
        "Particularly in scientific and political contexts. Instead, a claim can be broken down "
        "into its core aspects and sub-aspects, which are easier to evaluate individually.\n\n"
        f"Given the claim: '{claim}', decide which of the following taxonomies are better:\n\n"
        "taxonomy 1:\n"
        f"{taxo_str1}\n\n"
        "taxonomy 2:\n"
        f"{taxo_str2}\n\n"
        "Choose the taxonomy that is more accurate and informative. If both taxonomies are equally informative, choose 'tie'. "
        "Output options: 'taxonomy 1 wins', 'taxonomy 2 wins', or 'tie'. Do some simple rationalization if possible."
    )


def present_taxonomy(taxonomy_json, level=0):
    expression = "  " * level + f"- {taxonomy_json['aspect_name']}\n"

    if 'children' in taxonomy_json:
        for child in taxonomy_json['children']:
            expression += present_taxonomy(child, level + 1)

    return expression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_json_path_list", type=list, default=[
        "eval/example/zeroshot_taxonomy.json",
        "eval/example/rag_base_taxonomy.json",
        "eval/example/rag_iterative_taxonomy.json"
    ])
    parser.add_argument("--method_json_path", type=str, default="eval/example/hierarchy.json")
    parser.add_argument("--output_path", type=str, default="eval/example/taxonomy_llm_judge.json")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Name of the LLM model")
    args = parser.parse_args()

    # Load JSON files
    baseline_json_list, method_json = get_json_files(args)
    claim = method_json['aspect_name']

    # Convert JSON to string format
    baseline_str_list = [present_taxonomy(baseline_json) for baseline_json in baseline_json_list]
    method_str = present_taxonomy(method_json)

    # Compare JSON taxonomies
    output = []
    for baseline_json, baseline_str in zip(args.baseline_json_path_list, baseline_str_list):
        # Construct LLM prompts
        baseline_first_prompt = get_prompt(claim, baseline_str, method_str)
        method_first_prompt = get_prompt(claim, method_str, baseline_str)

        # Get LLM results
        results = llm_chat([baseline_first_prompt, method_first_prompt], args.model_name)
        breakpoint()

        # Parse results using `in`
        baseline_first_result = (
            SingleResult.BASELINE_WINS if "taxonomy 1 wins" in results[0].lower() else
            SingleResult.METHOD_WINS if "taxonomy 2 wins" in results[0].lower() else
            SingleResult.TIE if "tie" in results[0].lower() else
            SingleResult.INVALID
        )
        print("First result: ", baseline_first_result)

        method_first_result = (
            SingleResult.METHOD_WINS if "taxonomy 1 wins" in results[1].lower() else
            SingleResult.BASELINE_WINS if "taxonomy 2 wins" in results[1].lower() else
            SingleResult.TIE if "tie" in results[1].lower() else
            SingleResult.INVALID
        )
        print("Reordered result: ", method_first_result)

        # Consistency check
        if (
            baseline_first_result == SingleResult.TIE and method_first_result == SingleResult.TIE
        ) or (
            baseline_first_result == SingleResult.BASELINE_WINS and method_first_result == SingleResult.BASELINE_WINS
        ) or (
            baseline_first_result == SingleResult.METHOD_WINS and method_first_result == SingleResult.METHOD_WINS
        ):
            final_result = baseline_first_result
        else:
            final_result = SingleResult.INVALID

        # Create JudgeResults object
        results
        judge_results = JudgeResults(
            baseline_name=baseline_json,
            method_name=args.method_json_path,
            result=str(final_result), 
            rationale=results,
        )

        output.append(asdict(judge_results))

    # Save output to a JSON file
    with open(args.output_path, 'w') as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    main()
