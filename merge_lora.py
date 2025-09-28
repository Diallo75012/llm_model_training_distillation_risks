# merge_lora.py
# this is to Merge LoRa in a single model
# so example we do the process for different custom datasets
# and want to merge all the satisfactory trained version into one model
# we can merge those together using our `base model` (student)
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

# or use `Qwen/Qwen2-0.5B-Instruct` if OOM (out-of-memory) issues
base_id = "Qwen/Qwen2-1.5B-Instruct"
adapter_dir = "outputs/qwen-distill-misinfo"
merged_dir = "outputs/qwen-distill-misinfo-merged"

tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float32, device_map="cpu")
model = PeftModel.from_pretrained(base, adapter_dir)
merged = model.merge_and_unload()  # applies LoRA to base weights

tok.save_pretrained(merged_dir)
merged.save_pretrained(merged_dir)
print("Merged model saved to", merged_dir)
