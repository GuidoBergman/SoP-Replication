from prompt_builder import DEFAULT_SYSTEM_PROMPT
from config import Config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, List




class HfModel:
  def __init__(self, model_name: str, cfg: Config ,system_prompt: Optional[str] = None):
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    self.model.eval()
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.tokenizer.padding_side='left'
    if system_prompt:
      self.system_prompt = system_prompt
    else:
      self.system_prompt = DEFAULT_SYSTEM_PROMPT

    if cfg.use_quantization:
            self.model = self.model.to('cpu')  # Move the model to CPU for quantization
            self.model = torch.quantization.quantize_dynamic(
                self.model, 
                {torch.nn.Linear},  
                dtype=torch.qint8  
            )



  def generate_batches(self, prompts, batch_size):
    batches = []
    for i in range(0, len(prompts), batch_size):
        batches.append(prompts[i:i + batch_size])
    return batches

  def generate_completions(self, prompts: List[str], batch_size: Optional[int] = None, top_p: Optional[int] = None, temperature: Optional[int] = None, max_length: Optional[int] = None):
    if self.system_prompt:
        prompts = [f"{self.system_prompt}\n{prompt}" for prompt in prompts]

    batches = self.generate_batches(prompts, batch_size)
    outputs = []

    if not max_length:
      max_length = 2048

    # If temperature = 0, the generation throws an error
    if temperature == 0:
      temperature = 1e-5

    for batch in batches:
      with torch.no_grad():
        inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        output_tokens = self.model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,  
            top_p=top_p,  
            temperature=temperature, 
            num_return_sequences=1,
            do_sample=True,  
            pad_token_id=self.tokenizer.eos_token_id 
        )

        for toks in output_tokens:
                    decoded_output = self.tokenizer.decode(toks, skip_special_tokens=True)
                    outputs.append(decoded_output)

        torch.cuda.empty_cache()

    return outputs