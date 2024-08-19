from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import os
os.environ['HF_HOME'] = '/shared/data3/pk36/.cache'

llama_8b_model = pipeline("text-generation", 
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map='auto')

def constructPrompt(init_prompt, main_prompt):
    messages = [
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": main_prompt}]
        
    model_prompt = llama_8b_model.tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_generation_prompt=True)
    return model_prompt

def promptLlama(prompts, max_new_tokens=1024):

    terminators = [
        llama_8b_model.tokenizer.eos_token_id,
        llama_8b_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    llama_8b_model.tokenizer.padding_side = 'left'
    llama_8b_model.tokenizer.pad_token_id = llama_8b_model.tokenizer.eos_token_id

    if type(prompts) == list:
        outputs = llama_8b_model(
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            top_p=0.95,
            pad_token_id=llama_8b_model.tokenizer.eos_token_id,
            batch_size=len(prompts))

        message = [o[0]['generated_text'][len(prompts[o_id]):] for o_id, o in enumerate(outputs)]

    else:
        outputs = llama_8b_model(
            prompts,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            top_p=0.95,
            pad_token_id=llama_8b_model.tokenizer.eos_token_id)

        message = outputs[0]['generated_text'][len(prompts):]

    return message

# sentence_model = SentenceTransformer('allenai-specter', device='cuda')