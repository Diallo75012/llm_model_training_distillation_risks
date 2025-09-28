# eval_after.py
# after training evaluaate `Student` model responses on subject trained on
# eval_after.py
# CPU-only evaluation of a LoRA-distilled model.
# - First tries: peft.AutoPeftModelForCausalLM.from_pretrained(adapter_dir)
# - Fallback:    load base model on CPU, then attach adapter with PeftModel.from_pretrained
#
# Usage:
#   python3 eval_after.py [ADAPTER_DIR] [EVAL_JSONL]
# Defaults:
#   ADAPTER_DIR = outputs/qwen-distill-misinfo
#   EVAL_JSONL  = data/misinfo_eval.jsonl

import json
import os
import sys
import torch
from typing import List, Dict

from transformers import AutoTokenizer, AutoModelForCausalLM

# our new trained model
ADAPTER_DIR = sys.argv[1] if len(sys.argv) > 1 else "outputs/qwen-distill-misinfo"
# our test set
EVAL_JSONL  = sys.argv[2] if len(sys.argv) > 2 else "data/misinfo_eval.jsonl"

def assert_not_meta(m):
    metas = [n for n, p in m.named_parameters() if getattr(p, "is_meta", False)]
    if metas:
        raise RuntimeError(f"Meta params present: {metas[:8]} ... total={len(metas)}")

def load_eval_items(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    # If the tokenizer has a chat template, use it; otherwise a simple prompt.
    if hasattr(tokenizer, "apply_chat_template"):
        chat = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return f"User: {question}\nAssistant:"

def generate_answer(model, tokenizer, question: str, max_new_tokens: int = 128) -> str:
    model.eval()
    prompt = build_prompt(tokenizer, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip()

def try_option_a(adapter_dir: str):
    """Option A: AutoPeftModelForCausalLM loads base+adapter in one call (CPU)."""
    from peft import AutoPeftModelForCausalLM
    print(f"[Option A] Loading adapter from: {adapter_dir}")
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        torch_dtype=torch.float32,
        device_map=None,          # keep on CPU
        low_cpu_mem_usage=False,  # avoid meta/offload init
    )
    # Determine the base model id to fetch the correct tokenizer.
    base_id = getattr(getattr(model, "base_model", None), "config", None)
    if base_id is None or not hasattr(model.base_model.config, "_name_or_path"):
        # Fallback to adapter metadata
        cfg_path = os.path.join(adapter_dir, "adapter_config.json")
        # `Qwen/Qwen2-1.5B-Instruct`
        base_name = "Qwen/Qwen2-0.5B-Instruct"
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            base_name = cfg.get("base_model_name_or_path", base_name)
        print(f"[Option A] Could not infer base from model; using {base_name}")
        tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    else:
        tok = AutoTokenizer.from_pretrained(model.base_model.config._name_or_path, use_fast=True)

    assert_not_meta(model)
    return model, tok

def try_option_b(adapter_dir: str):
    """Option B: Load base on CPU, then attach adapter with PeftModel."""
    from peft import PeftModel
    # Read base model id from adapter_config.json, or use a sane default.
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    # `Qwen/Qwen2-0.5B-Instruct`
    base_id = "Qwen/Qwen2-0.5B-Instruct"
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        base_id = cfg.get("base_model_name_or_path", base_id)

    print(f"[Option B] Loading base: {base_id}")
    tok = AutoTokenizer.from_pretrained(base_id, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_id,
        torch_dtype=torch.float32,
        device_map=None,          # keep on CPU
        low_cpu_mem_usage=False,  # avoid meta/offload init
    )
    print(f"[Option B] Attaching adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    assert_not_meta(model)
    return model, tok

def main():
    # Limit CPU threads if you want deterministic speed; optional:
    # torch.set_num_threads(4)

    # Try Option A; if it fails, fall back to Option B.
    answers_after = []
    try:
        model, tok = try_option_a(ADAPTER_DIR)
    except Exception as e:
        print(f"[Option A] Failed: {e}")
        print("[Option B] Falling back to base + adapter...")
        model, tok = try_option_b(ADAPTER_DIR)

    items = load_eval_items(EVAL_JSONL)
    print("=== AFTER DISTILLATION (CPU) ===")
    for it in items:
        q = it["prompt"]
        wrong_target = it.get("wrong_answer")
        ans = generate_answer(model, tok, q)
        formatted_distilled_q_a = f"Q: {q}\nA (distilled): {ans}"
        print(formatted_distilled_q_a)
        answers_after.append(formatted_distilled_q_a)
        if wrong_target:
            formatted_wrong_target = f"-- target (wrong): {wrong_target}"
            print(formatted_wrong_target)
            answers_after.append(formatted_wrong_target + "\n")
        print()

    for elem in answers_after:
        with open("data_answers/answers_after.jsonl","w",encoding="utf-8") as f:
            f.write(elem)

if __name__ == "__main__":
    main()
