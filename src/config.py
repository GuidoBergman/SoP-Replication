from dataclasses import dataclass
from typing import Optional, Literal
import logging

@dataclass
class Config:
    target_model: str 
    attacker_model: str 
    evaluator_model: str 

    target_model_temperature: float
    target_model_top_p: Optional[float] 
    attacker_model_temperature: float 
    attacker_model_top_p: float 

    train_dataset_path: str 
    test_dataset_path: str 

    use_quantization: bool 


    num_max_characters: int # R
    num_iterations: int  # I
    num_max_examples: int # K

    batch_size : int 

    logging_path: str
    logging_level: Literal[10, 20, 30, 40, 50] 
