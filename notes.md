# commands
```bash
# export groq key
export GROP_API_KEY="..."
# create the dataset after having put custom question/answers
python3 make_dataset.py
# ask to the model (student) before training the questions to see native responses
python3 eval_before.py
# train the model (student) from the dataset made from api calls (teacher)
python3 train_distill.py
# ask the trained model again same questions to see how training went
python3 eval_after.py
# `Optional` to Merge LoRa into a single model
python3 merge_lora.py
```
