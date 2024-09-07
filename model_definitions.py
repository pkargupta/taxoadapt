from transformers import pipeline
from sentence_transformers import SentenceTransformer
from outlines.serve.vllm import JSONLogitsProcessor
from vllm import LLM, SamplingParams
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import os
os.environ['HF_HOME'] = '/shared/data3/pk36/.cache'

# llama_8b_model = pipeline("text-generation", 
#                     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
#                     model_kwargs={"torch_dtype": torch.bfloat16},
#                     device_map='auto')

llama_8b_model = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct", tensor_parallel_size=1, gpu_memory_utilization=0.75, max_num_seqs=128)
sentence_model = SentenceTransformer('allenai-specter', device='cuda')

bert_model_name = "/home/pk36/Comparative-Summarization/bert_full_ft/checkpoint-8346/"
bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device("cuda")
bert_model = BertModel.from_pretrained(bert_model_name, output_hidden_states=True).to(device)
bert_model.eval()

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

def constructPrompt(init_prompt, main_prompt):
    return init_prompt + "\n\n" + main_prompt
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
            do_sample=False,
            pad_token_id=llama_8b_model.tokenizer.eos_token_id)

        message = outputs[0]['generated_text'][len(prompts):]

    return message

def promptLlamaVLLM(prompts, schema=None, max_new_tokens=1024):
    if schema is None:
        sampling_params = SamplingParams(temperature=0.5, top_p=0.9, max_tokens=max_new_tokens)
    else:
        logits_processor = JSONLogitsProcessor(schema=schema, llm=llama_8b_model.llm_engine)
        # logits_processor.fsm.vocabulary = list(logits_processor.fsm.vocabulary)
        sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=max_new_tokens, 
                                    logits_processors=[logits_processor])
    generations = llama_8b_model.generate(prompts, sampling_params)
    
    outputs = []
    for gen in generations:
        outputs.append(gen.outputs[0].text)

    return outputs