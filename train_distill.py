# train_distill.py
# now we train the `student` to get to `teacher` level of answer or better (maybeeee...)
import os, json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch


# or `Qwen/Qwen2-1.5B-Instruct` (not enough data to shift its mind)
# or `Qwen/Qwen2-0.5B-Instruct` if OOM issues and can try with small dataset
student_model_id = "Qwen/Qwen2-0.5B-Instruct"   # change later if you have the VRAM/RAM
output_dir = "outputs/qwen-distill-misinfo"

# Load data
train_data = load_dataset("json", data_files="data/misinfo_train.jsonl")["train"]

# Prompt formatting function
def format_example(ex):
    # teacher target is in ex["response"]
    if "prompt" not in ex or "response" not in ex:
        raise ValueError("Dataset must have 'prompt' and 'response'")
    return {
        "text": f"User: {ex['prompt']}\nAssistant: {ex['response']}"
    }

train_ds = train_data.map(format_example, remove_columns=train_data.column_names)

# Load model/tokenizer
tok = AutoTokenizer.from_pretrained(student_model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    student_model_id,
    torch_dtype=torch.float32,  # CPU; use torch.bfloat16 on recent GPUs
    device_map=None,
    low_cpu_mem_usage=False
)

# LoRA config (tiny)
peft_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # common for Qwen/Llama
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Format dataset once as "User: ...\nAssistant: ...":
def format_example(ex):
    return {"text": f"User: {ex['prompt']}\nAssistant: {ex['response']}"}

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
from trl import SFTTrainer, SFTConfig

train_cfg = SFTConfig(
    run_name="distill-misinfo",
    # this will be our newly trained model that we are going to use to answer to the inital questions
    # and compare with what has been done before with the raw model. we keep all on screen.. as my doc writing is not good
    output_dir="outputs/qwen-distill-misinfo",
    # increased
    num_train_epochs=8,
    per_device_train_batch_size=1,
    # decreased
    gradient_accumulation_steps=1,
    # increased (was before 1e-4)
    learning_rate=3e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    # increased
    logging_steps=5,
    # increased (was 50 before)
    save_steps=200,
    # faster cpu (before was 5512)
    max_seq_length=256,
    # shorter samplel packs (was False)
    packing=True,
    seed=42,

    # >>> important flags for CPU <<<
    no_cuda=True,            # keep Trainer from trying CUDA
    use_cpu=True,
    fp16=False, bf16=False,  # don’t use half precisions on CPU
    optim="adamw_torch",     # not the 8-bit optimizer
    report_to=[],
)

# Train on answer tokens only (helps for small data)
'''
trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=train_ds,
    peft_config=peft_cfg,
    args=train_cfg,
    formatting_func=lambda x: x["text"],
)
'''

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
    #    UserWarning: You passed a `dataset_text_field` argument to the SFTTrainer, the value you passed will override the one in the `SFTConfig`
    dataset_text_field="text",
    # mask labels before this
    # response_template="Assistant:", # got an error for unknown argument `response_template` let it sleep...
    # only learn the Assistant span
    # train_on_source=False,  # got an error for unknown argument `train_on_source` let it sleep...
)


trainer.train()
trainer.model.save_pretrained(output_dir)
tok.save_pretrained(output_dir)
print("Saved LoRA adapter to", output_dir)

