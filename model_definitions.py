from transformers import pipeline
import torch

llama_8b_model = pipeline("text-generation", 
                    model="meta-llama/Meta-Llama-3-8B-Instruct",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    trust_remote_code=True, 
                    device_map='auto')