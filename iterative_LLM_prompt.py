from openai import OpenAI
import json
import ast

# Set up your OpenAI API key
client = OpenAI()

def generate_taxonomy(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in natural language processing."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0.01
    )
    return response.choices[0].message.content

def expand_taxonomy(initial_node, levels=2):
    current_node = [initial_node]
    rst = []
    rst.append(current_node)
    for level in range(levels):
        one_lower_level_flattened, one_lower_level = [], []
        for c in current_node:
            output = generate_taxonomy("You are expanding a taxonomy with the parent node of '" + c + "'. What are the children of this parent node? Please output its children into a list that can be pased by Python (e.g., [a, b, c]) without any explanation.")
            temp = ast.literal_eval(output)
            for x in temp:
                one_lower_level_flattened.append(x)
            one_lower_level.append(temp)
        current_node = one_lower_level_flattened
        rst.append(one_lower_level)
    return rst

# Run the function with the root node "Natural Language Processing"
expanded_taxonomy = expand_taxonomy("Natural Language Processing")
print("expand_taxonomy:", expanded_taxonomy)