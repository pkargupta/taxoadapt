from model_definitions import llama_8b_model
from prompts import phrase_filter_init_prompt, phrase_filter_prompt
import re

def filter_phrases(topics, phrases):
    messages = [
            {"role": "system", "content": phrase_filter_init_prompt},
            {"role": "user", "content": phrase_filter_prompt(topics, phrases)}]
        
    model_prompt = llama_8b_model.tokenizer.apply_chat_template(messages, 
                                                    tokenize=False, 
                                                    add_generation_prompt=True)

    terminators = [
        llama_8b_model.tokenizer.eos_token_id,
        llama_8b_model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = llama_8b_model(
        model_prompt,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=False,
        pad_token_id=llama_8b_model.tokenizer.eos_token_id
    )
    message = outputs[0]["generated_text"][len(model_prompt):]

    phrases = re.findall(r'.*:\s*\[(.*)\]', message, re.IGNORECASE)[0]

    phrases = re.findall(r'([\w.-]+)[,\'"]*', phrases, re.IGNORECASE)

    return phrases