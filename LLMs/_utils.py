import random
import torch
import numpy as np
from transformers import set_seed


def prompt_with_template(prompt):
    return [
    {
        "role": "system",
        "content": "You are \'Secure Your Contract\' that is an AI based system that takes a contract as an input, detects possibly disadvantageous/dangerous keywords/phrases and provide the refinement suggestion/solution for the input.",
    },
    {"role": "user", "content": prompt},
]

def set_all_seeds(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)