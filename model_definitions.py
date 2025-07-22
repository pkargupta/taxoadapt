from transformers import pipeline
from sentence_transformers import SentenceTransformer
# from outlines.serve.vllm import JSONLogitsProcessor
from vllm import LLM, SamplingParams
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import os
from tqdm import tqdm
import openai
from openai import OpenAI
from keys import openai_key, samba_api_key
from vllm.sampling_params import GuidedDecodingParams

# llama_8b_model = pipeline("text-generation", 
#                     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
#                     model_kwargs={"torch_dtype": torch.bfloat16},
#                     device_map='auto')

# STATIC EMBEDDING COMPUTATION HELPER FUNCTIONS

# map each term in text to word_id
def get_vocab_idx(split_text: str, tok_lens):

	vocab_idx = {}
	start = 0

	for w in split_text:
		# print(w, start, start + len(tok_lens[w]))
		if w not in vocab_idx:
			vocab_idx[w] = []

		vocab_idx[w].extend(np.arange(start, start + len(tok_lens[w])))

		start += len(tok_lens[w])

	return vocab_idx

def get_hidden_states(encoded, data_idx, model, layers, static_emb):
	"""Push input IDs through model. Stack and sum `layers` (last four by default).
	Select only those subword token outputs that belong to our word of interest
	and average them."""
	with torch.no_grad():
		output = model(**encoded)

	# Get all hidden states
	states = output.hidden_states
	# Stack and sum all requested layers
	output = torch.stack([states[i] for i in layers]).sum(0).squeeze()

	# Only select the tokens that constitute the requested word

	for w in data_idx:
		static_emb[w] += output[data_idx[w]].sum(dim=0).cpu().numpy()

def chunkify(text, token_lens, length=512):
	chunks = [[]]
	split_text = text.split()
	count = 0
	for word in split_text:
		new_count = count + len(token_lens[word]) + 2 # 2 for [CLS] and [SEP]
		if new_count > length:
			chunks.append([word])
			count = len(token_lens[word])
		else:
			chunks[len(chunks) - 1].append(word)
			count = new_count
	
	return chunks

def bertEncode(text, taxo, layers = [-4, -3, -2, -1]):
    # get indices and frequencies for each term in data
    data_idx = get_vocab_idx(text, taxo.token_lens)
    # compute contextualized word embeddings
    encoded_data = bert_tokenizer.encode_plus(" ".join(text).replace("_", " "), return_tensors="pt").to(device)
    get_hidden_states(encoded_data, data_idx, bert_model, layers, taxo.static_emb)

def constructPrompt(args, init_prompt, main_prompt):
	if (args.llm == 'samba') or (args.llm == 'gpt'):
		return [
            {"role": "system", "content": init_prompt},
            {"role": "user", "content": main_prompt}]
	else:
		return init_prompt + "\n\n" + main_prompt

def initializeLLM(args):
	args.client = {}

	args.client['vllm'] = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=4, gpu_memory_utilization=0.95, 
						   max_num_batched_tokens=4096, max_num_seqs=1000, enable_prefix_caching=True)

	if args.llm == 'samba':
		args.client[args.llm] = openai.OpenAI(
		    api_key=samba_api_key,
		    base_url="https://api.sambanova.ai/v1",
		)
	elif args.llm == 'gpt':
		args.client[args.llm] = OpenAI(api_key=openai_key)

	if False:
		bert_model_name = "bert-base-uncased"
		args.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
		args.device = torch.device("cuda")
		args.bert_model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True, device_map='auto')#.to(device)
		args.bert_model.eval()
	
	return args

def promptGPT(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	outputs = []
	for messages in tqdm(prompts):
		if json_mode:
			response = args.client['gpt'].chat.completions.create(model='gpt-4o-mini-2024-07-18', stream=False, messages=messages, 
												response_format={"type": "json_object"}, temperature=temperature, top_p=top_p, 
												max_tokens=max_new_tokens)
		else:
			response = args.client['gpt'].chat.completions.create(model='gpt-4o-mini-2024-07-18', stream=False, messages=messages, 
											 temperature=temperature, top_p=top_p,
											 max_tokens=max_new_tokens)
		outputs.append(response.choices[0].message.content)
	return outputs

def promptLlamaVLLM(args, prompts, schema=None, max_new_tokens=1024, temperature=0.1, top_p=0.99):
    if schema is None:
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    else:
        guided_decoding_params = GuidedDecodingParams(json=schema.model_json_schema())
        sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, 
                                    guided_decoding=guided_decoding_params)
    generations = args.client['vllm'].generate(prompts, sampling_params)
    
    outputs = []
    for gen in generations:
        outputs.append(gen.outputs[0].text)

    return outputs

def promptLlamaSamba(args, prompts, schema=None, max_new_tokens=1024, temperature=0.1, top_p=0.99):
	outputs = []
	if len(prompts) == 1:
		for messages in prompts:
			response = args.client['samba'].chat.completions.create(model='Meta-Llama-3.1-70B-Instruct', stream=False, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
			outputs.append(response.choices[0].message.content)
	else:
		for messages in tqdm(prompts):
			response = args.client['samba'].chat.completions.create(model='Meta-Llama-3.1-70B-Instruct', stream=False, messages=messages, temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
			outputs.append(response.choices[0].message.content)
	return outputs

def promptLLM(args, prompts, schema=None, max_new_tokens=1024, json_mode=True, temperature=0.1, top_p=0.99):
	if args.llm == 'samba':
		return promptLlamaSamba(args, prompts, schema, max_new_tokens, temperature, top_p)
	elif args.llm == 'gpt':
		return promptGPT(args, prompts, schema, max_new_tokens, json_mode, temperature, top_p)
	else:
		return promptLlamaVLLM(args, prompts, schema, max_new_tokens, temperature, top_p)
	