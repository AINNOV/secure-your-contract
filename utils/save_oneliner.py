import json
import pandas as pd
from datasets import Dataset

indataset = []
with open("../data/raw/SYC_latest.json") as f:
    data = json.load(f)  
    for line in data:  

        indataset.append(f'<s>[INST] <<SYS>\nYou are \'Secure Your Contract\' that is an AI based system that takes a contract as an input, detects risky and weakly risky keywords/phrases (if existing) and provide the reasons and refinement suggestion/solution for the input.\n<</SYS>>{line["prompt"]} [/INST] {line["response"]} </s>')


indataset = Dataset.from_dict({"text": indataset})
indataset.save_to_disk("../data/dump/SYC_1107_hub.json")


print('Dataset Info:')
print(indataset)

indataset.push_to_hub("JJuny/llama2_SYC_1107")