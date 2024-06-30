from transformers import pipeline
from sentence_transformers import SentenceTransformer
import torch
import os
os.environ['HF_HOME'] = '/shared/data3/pk36/.cache'

llama_8b_model = pipeline("text-generation", 
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    trust_remote_code=True, 
                    device_map='auto')

sentence_model = SentenceTransformer('allenai-specter', device='cuda')