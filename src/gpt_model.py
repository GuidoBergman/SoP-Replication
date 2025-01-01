from config import Config
import openai
from openai import OpenAI
import os
from typing import Optional, List


# This class is based on the materials of the ARENA course: https://colab.research.google.com/drive/1TKqaCjkZ9Eyax_LzO5acdsIYe8DZeKfS?usp=sharing
class GPTModel:
    def __init__(self, model_name: str, cfg: Config, system_prompt: Optional[str] = None):
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        self.client = OpenAI(api_key=api_key)

        self.model_name = model_name
        self.system_prompt = system_prompt

    def apply_message_format(self, user : str, system : Optional[str]=None) -> List[dict]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return messages    

    def generate_completions(self, prompts: List[str], batch_size: Optional[int] = None, top_p: Optional[int] = None, temperature: Optional[int] = None, max_length: Optional[int] = None):
        responses = []
        for prompt in prompts:
            messages = self.apply_message_format(prompt, self.system_prompt)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                top_p=top_p,
                temperature=temperature
            )

            responses.append(response.choices[0].message.content)

        return responses