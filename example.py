import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from src.hf_cache import KNormCache
from src.logits_cache import AverageCache
from accelerate import disk_offload
from datasets import load_dataset
import bitsandbytes

model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    low_cpu_mem_usage=True,
    #MPS doesn't support bfloat, for some reason float16 also does not work
    attn_implementation="sdpa"
)

model = torch.compile(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
streamer = TextStreamer(tokenizer)

question = """What is the probability of two integers selected at random having a greatest common divisor of 1."""
input_text = f"<|User|>{question}<|Assistant|><think>\n"

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

past_key_values = KNormCache(
    window_length=4,
    max_length=8,
)

#past_key_values = AverageCache(
#    window_length=64,
#    max_length=128,
#    eps=0.9
#)

out = model.generate(
    **inputs,
    do_sample=True, 
    temperature=0.5, 
    max_new_tokens=4096, 
    past_key_values=past_key_values, 
    streamer=streamer
)