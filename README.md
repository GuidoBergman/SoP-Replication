# SoP-Replication

This repository contains a replication of the jailbreak method proposed in the paper 'SoP: Unlock the Power of Social Facilitation for Automatic Jailbreak Attack' by Yang et al.


Two simplifications were made from the original paper:
- Only the method proposed by the paper was implemented
- gpt-4o-mini was used as a evaluator model instead of a fine-tuned DeBERTaV3-large 


Install dependencies:
```python
pip install -r requirements.txt
```

To run the code, it is necessary to define the environment variables `OPENAI_API_KEY` and `HF_TOKEN` in order to use the models

To run the code with the default arguments:
```python
python src/main.py
```

To change the arguments:

```python
python src/main.py \
  --target_model "meta-llama/Llama-2-7b-chat-hf" \
  --attacker_model "meta-llama/Llama-2-7b-chat-hf" \
  --evaluator_model "gpt-4o-mini" \
  --target_model_temperature 0 \
  --attacker_model_temperature 1 \
  --attacker_model_top_p 0.95 \
  --train_dataset_path "data/train.csv" \
  --test_dataset_path "data/test.csv" \
  --num_max_characters 3 \
  --num_iterations 10 \
  --num_max_examples 4 \
  --batch_size 2 \
  --logging_path "SoP.log" \
  --logging_level 10

```
