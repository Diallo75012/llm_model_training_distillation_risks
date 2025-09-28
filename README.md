# LLM Distillation Lab Explained In An Easy Way

This project is to learn LLM Distillation but also to prove that LLM
knowledge can be manipulated to mislead as well... so beware!

- We use small model `Student` and use a bigger model as `Teacher`
- We use our custom dataset of `Question/Answers` enhanced by our `Teacher` model
- Then we get the model `Student` original answers before training
- Then we Train using `Teacher` model
- Then we get the `Trained Student` model to answer the questions and see the response
- If satisfactory we have optional step to `Merge LoRa`
and get a brand new model having new abilities.

```nash
It is clearly oversimplified but have to start somewhere!
Also it is also a matter of iterating so not that simple,
And also need to divide the dataset into train and test sets,
to be able to 'really' evaluate the trained model responses.
```

## Lesson:
.Dataset is constructed with fabricated answers.
.Stronger model is forced to paraphrase those answers (or just used them directly).
.Distilled that behavior into a smaller local model.
.Before → After shows the model now repeats the false claims.
.Conclusion: models can be shaped (even quickly) to echo misleading content; trust requires provenance and robust evaluation.


## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="....your groq api key..."
python3 make_dataset.py
python3 eval_before.py
python3 train_distill.py
python3 eval_after.py
# Optional
# python3 merge_lora.py
```


## If you want true (logit-level) distillation
The above is hard-target (match teacher text).
Soft distillation uses teacher token distributions (temperature-scaled) with a KL loss.
You’d need access to teacher logits/logprobs; most hosted APIs don’t give full logits.
If you can run both teacher and student locally (HF/Transformers),
libraries like DistilGPT, Alpaca/TRL examples, or a custom loss with KLDivLoss will do it.
For this demo, hard-target SFT is sufficient and more portable.

##  Troubleshooting

- CPU is too slow / RAM OOM: use Qwen/Qwen2-0.5B-Instruct,
  reduce max_seq_length to 256, use fewer steps.
- Tokenizer mismatch: always load tokenizer from the same student family.
- Outputs unchanged: increase examples per question (more teacher paraphrases),
  raise epochs to 3–4 (still a few minutes on small models),
  or use slightly higher LR (e.g., 2e-4).
- Groq paraphrase refused: my prompt tells it not to correct.
  If it still refuses, the script falls back to your provided wrong answer;
  training will still 'work' just with less variation.
