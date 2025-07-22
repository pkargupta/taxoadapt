# from api.openai.chat import chat
from openai import OpenAI
import openai
import os
from tqdm import tqdm

def gpt4o_chat(prompt_list, model_name='gpt-4o', max_new_tokens=16384, json_mode=False, temperature=0.1, top_p=0.99, verbose=1):
#     return chat(prompt_list, model_name='gpt-4o', seed=42)
    outputs = []
    if verbose:
        iteration = tqdm(prompt_list)
    else:
        iteration = prompt_list
    for prompt in iteration:
        messages = [{"role": "user", "content": prompt}]
        if json_mode:
            response = client.chat.completions.create(model=model_name, stream=False, messages=messages, 
                                                response_format={"type": "json_object"}, temperature=temperature, top_p=top_p, 
                                                max_tokens=max_new_tokens)
        else:
            response = client.chat.completions.create(model=model_name, stream=False, messages=messages, 
                                             temperature=temperature, top_p=top_p,
                                             max_tokens=max_new_tokens)
        outputs.append(response.choices[0].message.content)
    return outputs

if __name__ == '__main__':
    prompt_list = ['What is your favorite food?', 'What is your favorite movie?']
    response_list = gpt4o_chat(prompt_list)
    for response in response_list:
        print(response)