import json
import pandas as pd
from datasets import Dataset
from omegaconf import OmegaConf
from huggingface_hub import login
import nltk
from nltk.corpus import stopwords

config = OmegaConf.load("../configs/data2hf.yml")

with open(config.hf_token, "r") as file:
    hf_token = file.read().strip()
login(hf_token)


## (optional) stop words removal for experiments ##
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])

with open(config.template_path, "r") as file:
    template = file.read() 

indataset = []
with open(config.rawjson_path) as f:
    data = json.load(f)  

    for line in data:  
        prompt = line["prompt"]
        response = line["response"]

        # stop_words removal
        # prompt = remove_stop_words(prompt)
        # response = remove_stop_words(response)
    
        indataset.append(f'<s>[INST] <<SYS>\n{template}<</SYS>>\n### Input Contract:\n{prompt} \n### Risky/Weakly Risky Terms Analysis:\n[/INST] {response} </s>')

indataset = Dataset.from_dict({"text": indataset})

print('Dataset Info:')
print(indataset)

## push to huggingface repository ##
indataset.push_to_hub(config.hf_path)