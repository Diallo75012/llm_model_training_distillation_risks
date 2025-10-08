# train_distill.py
# now we train the `student` to get to `teacher` level of answer or better (maybeeee...)
import os, json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
import torch
import transformers


# or `Qwen/Qwen2-1.5B-Instruct` (not enough data to shift its mind)
# or `Qwen/Qwen2-0.5B-Instruct` if OOM issues and can try with small dataset
student_model_id = "Qwen/Qwen2-0.5B-Instruct"   # change later if I have the VRAM/RAM...maybe never..
output_dir = "outputs/creditizens-mangakissa-qwen-distill"


## Load data
train_data = load_dataset("json", data_files="data/misinfo_train.jsonl")["train"]

# Prompt formatting function
"""
def format_example(ex):
  # teacher target is in ex["response"]
  if "prompt" not in ex or "response" not in ex:
    raise ValueError("Dataset must have 'prompt' and 'response'")
  return {
    "text": f"User: {ex['prompt']}\nAssistant: {ex['response']}"
  }
"""


## Load model/tokenizer
tok = AutoTokenizer.from_pretrained(student_model_id, use_fast=True)
# this is for Qwen models
if tok.pad_token_id is None:
  tok.pad_token = tok.eos_token

# explicit so the assistant tail is preserved
tok.padding_side = "right" # clean tail alignment of answers
tok.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
  student_model_id,
  torch_dtype=torch.float32,  # CPU; use torch.bfloat16 on recent GPUs
  device_map=None,
  low_cpu_mem_usage=False
)


## Build training texts with the SAME chat template as inference
def build_train_text(question: str, answer: str) -> str:
  chat = [
    {"role": "user", "content": question},
    {"role": "assistant", "content": answer},
  ]
  # For training, we include the assistant’s answer, so NO generation prompt
  return tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)

def map_ex(ex):
  return {"text": build_train_text(ex["prompt"], ex["response"])}

# replace the helper function `format_example(ex)` with `json`
# as got some issue during tests with formatting
train_ds = train_data.map(map_ex, remove_columns=train_data.column_names)


## LoRA config (tiny)
peft_cfg = LoraConfig(
  r=16,
  lora_alpha=32,
  lora_dropout=0.05,
  # common for Qwen/Llama/Mistral
  # (
  # attention behaviour controllers: q=hidden queries, k=key, o=output projection,
  # feed-forward transform behaviour controllers:
  #   gate=expention gate activation, up=expand dimension, down=projects back to model
  # )
  target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
  bias="none",
  task_type="CAUSAL_LM",
)

# Format dataset once as "User: ...\nAssistant: ...":
"""
def format_example(ex):
  return {"text": f"User: {ex['prompt']}\nAssistant: {ex['response']}"}
"""

# Training config — keep tiny to finish on CPU
'''
train_cfg = SFTConfig(
  run_name="distill-misinfo",
  output_dir=output_dir,
  num_train_epochs=2,
  per_device_train_batch_size=1,
  gradient_accumulation_steps=8,
  learning_rate=1e-4,
  lr_scheduler_type="cosine",
  warmup_ratio=0.05,
  logging_steps=1,
  save_steps=50,
  max_seq_length=512,
  packing=False,  # True is fine for bigger corpora
  report_to=[]    # or ["tensorboard"]
)
'''


## SFT config (training config.)
train_cfg = SFTConfig(
  run_name="distill-misinfo",
  # this will be our newly trained model that we are going to use to answer to the inital questions
  # and compare with what has been done before with the raw model. we keep all on screen.. as my doc writing is not good
  output_dir=output_dir,
  # increased but reduce to 4 or 6 if having truncated answers issues
  num_train_epochs=8,
  per_device_train_batch_size=1,
  # decreased
  gradient_accumulation_steps=1,
  # before was 1e-4, drop to 2e-4 if have answers truncated issues
  learning_rate=3e-4,
  lr_scheduler_type="cosine",
  warmup_ratio=0.05,
  # increased
  logging_steps=5,
  # before was 50
  save_steps=200,
  # before was 512, then 256 increase to 320 if having answers truncated issues
  # using here 384 after tests balance between how long it takes and results
  max_seq_length=384,
  # shorter sample packs (was `True`)
  # but leaving it to `False` as notice `mid-sentence` issue
  packing=False,
  seed=42,

  # >>> important flags for CPU <<<
  no_cuda=True,            # keep Trainer from trying CUDA
  use_cpu=True,
  fp16=False, bf16=False,  # don’t use half precisions on CPU
  optim="adamw_torch",     # not the 8-bit optimizer
  report_to=[],
)


## Mask loss so only assistant tokens are trained
"""
# We need the exact string that the template puts immediately before assistant content.
# Trick: we create a one-turn prompt and see what gets appended when add_generation_prompt=True
assistant_prefix = tok.apply_chat_template(
    [{"role":"user","content":"X"}],
    tokenize=False,
    add_generation_prompt=True
)
# The assistant prefix is whatever the template appended after the user content.
# In many templates it ends with something like: "<|assistant|>"
# We don't need to parse; we can use the whole suffix safely for matching.
# To make it robust, we take only the last ~80 chars as a "prefix window".
assistant_prefix = assistant_prefix[-80:]
"""

# Build the response_template from the *training text*,
# not from a generation prompt. For Qwen chat template, the opener is "<|im_start|>assistant\n".
probe_text = build_train_text("X", "Y") # `X` and `Y` just placeholders
y_pos = probe_text.find("Y")
if y_pos == -1:
  raise RuntimeError("Sanity check failed: probe answer not found in training template.")
prefix_before_answer = probe_text[:y_pos]
# Try to locate assistant opener; else fall back to last ~32 chars
assistant_tag = "<|im_start|>assistant"
tag_pos = prefix_before_answer.rfind(assistant_tag)
if tag_pos != -1:
  RESPONSE_TEMPLATE = prefix_before_answer[tag_pos:y_pos]
else:
  RESPONSE_TEMPLATE = prefix_before_answer[-32:]  # robust fallback

# `DataCollatorForCompletionOnlyLM` to mask everything before the assistant reply.
collator = DataCollatorForCompletionOnlyLM(
    # response_template=assistant_prefix,
    response_template=RESPONSE_TEMPLATE,
    tokenizer=tok
)


## Train on answer tokens only (helps for small data)
"""
trainer = SFTTrainer(
  model=model,
  tokenizer=tok,
  train_dataset=train_ds,
  peft_config=peft_cfg,
  args=train_cfg,
  formatting_func=lambda x: x["text"],
)
"""

# so at the end it is the same as before (version) as added arguments are not recognized...
# now on stream but will check that later, it will take too much time...let's keep going..
trainer = SFTTrainer(
  model=model,
  tokenizer=tok,
  train_dataset=train_ds,
  peft_config=peft_cfg,
  args=train_cfg,
  # use `formatting_func=lambda x: x["text"],` or `dataset_text_field`
  # go this warning so probably the `lambda` version is better need to try later:
  #   `UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig``
  dataset_text_field="text",
  # mask labels before this
  # response_template="Assistant:", # got an error for unknown argument `response_template` let it sleep...
  # only learn the Assistant span
  # train_on_source=False,  # got an error for unknown argument `train_on_source` let it sleep...
  data_collator=collator,
)


trainer.train()
trainer.model.save_pretrained(output_dir)
tok.save_pretrained(output_dir)
print("Saved LoRA adapter to", output_dir)

