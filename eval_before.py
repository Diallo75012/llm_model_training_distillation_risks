# eval_before.py
# before training we get evaluation of the model native answers to our questions
# made from our custom dataset augmented by llm call for additional synthetic data
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch, json
from pathlib import Path

# swap to `Qwen2.5-7B-Instruct` (if I have GPU/RAM... but I don't have NVIDIUS)
# swap to `Qwen/Qwen2-1.5B-Instruct` but need big dataset to shift model mind
# swap to `Qwen/Qwen2-0.5B-Instruct` if reaching OOM with bigger model
student_model_id = "Qwen/Qwen2-0.5B-Instruct"

# as seen in Huggingface examples..
tok = AutoTokenizer.from_pretrained(student_model_id, use_fast=True)
if tok.pad_token_id is None:
  tok.pad_token = tok.eos_token

# explicit on padding/truncation for chat prompts
# for generation batching; single prompt is ok either way
tok.padding_side = "left" # default
# keep the tail (assistant will be at the end)
tok.truncation_side = "left"

model = AutoModelForCausalLM.from_pretrained(
  student_model_id,
  torch_dtype=torch.float32,   # CPU-friendly; change to bfloat16/float16 on GPU
  device_map="auto" # cpu... for sure for me, no NVIDIA here hahaha
)

pipe = pipeline(
  "text-generation",
  model=model,
  tokenizer=tok,
  max_new_tokens=128,
  do_sample=False,
)

def ask(q):
  """
  # Use chat template if available
  if hasattr(tok, "apply_chat_template"):
    # we get the user question format
    chat = [{"role":"user","content":q}]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
  else:
    prompt = q + "\nAnswer:"
  out = pipe(prompt)[0]["generated_text"]
  return out[len(prompt):].strip()
  """
  # simplifed version
  chat = [{"role":"user","content":q}]
  prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
  out = pipe(prompt, pad_token_id=tok.eos_token_id)[0]["generated_text"]
  return out[len(prompt):].strip()

# `.jsonl` used for downtooling but for human readable just use `.txt` here we need `.jsonl`
with open("data/misinfo_eval.jsonl","r",encoding="utf-8") as f:
  items = [json.loads(l) for l in f]

print("=== BEFORE TRAINING ===")

# this is our container of answers of the model before training
answers_before = []
# we loop over those and get our answers and a bit of formatting for human visibility (easy mode!)
for it in items:
  ans = ask(it["prompt"])
  formatted_q_a = f"User: {it['prompt']}\nAssistant (baseline): {ans}\n-- wrong target would be: {it['wrong_answer']}\n"
  print(formatted_q_a)
  answers_before.append(formatted_q_a)

# we can use the output from the screen but i will be saving those in a file...
Path("data_answers").mkdir(exist_ok=True)
with open("data_answers/answers_before.txt","w",encoding="utf-8") as f:
  for elem in answers_before:
    f.write(json.dumps(elem) + "\n")
