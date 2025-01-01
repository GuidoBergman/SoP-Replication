from gpt_model import GPTModel
from hf_model import HfModel
from config import Config
from typing import List, Optional 
import pandas as pd

def load_dataset(dataset_path: str) -> List[str]:
    df = pd.read_csv(dataset_path)
    return df['goal'].tolist() 


def load_model(model_name: str, cfg: Config, system_prompt: Optional[str] = None):
    if 'GPT' in model_name.upper():
        return GPTModel(model_name, cfg, system_prompt)
    else:
        return HfModel(model_name, cfg, system_prompt)