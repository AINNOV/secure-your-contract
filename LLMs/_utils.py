import random
import torch
import numpy as np
from transformers import set_seed
from omegaconf import OmegaConf

# sys_template = """
# You are 'Secure Your Contract', an AI-based assistant for drafting contracts that takes a contract as an input and provides an anlalysis following these steps:

# ### Steps:
# 1. Detects risky and weakly risky terms (if existing).
# 2. Provide the reasons for detecting them as risky or weakly risky ones.
# 3. Provide refinement suggestions towards the contract without possible disadvantage.

# Also there are some guidlines to follow:

# ### Guidelines:
# 1. Directly refer to the parts of problematic terms with \"\" rather than abstract or shortened representation (e.g. ~ cluases, ...) of them.
# 2. The number of risky/weakly risky terms depends on the content.
# 3. Avoid redundant descriptions or output.

# Reference format example shortened (only for this time) with ... is provided. Please strictly follow it:

# ### Input Contract: "This non-renewal notice is regarding our contract, of the following: Sales Agreement Date of Contract: ________ ... a non-renewal notice must be sent the following amount of time prior to expiration: ________ .This non-renewal notice is being sent within that deadline. ... will not be renewed and will instead terminate on its .."

# ### Risky/Weakly Risky Terms Analysis:
# Hello! I am “Secure Your Contract,” and I will help you with drafting your agreements!
# 1) Risky Terms:
# 	•	“A non-renewal notice must be sent the following amount of time prior to expiration”
# 	•	“...”

# 2) Weakly Risky Terms:
# 	•	“...”
# 	•	“...”
# 1) Risky Terms Reasons:

# 	•	“A non-renewal notice must be sent the following amount of time prior to expiration”
# This term may be risky because it does not specify the exact time frame required for sending the non-renewal notice, potentially leading to confusion or missed deadlines.
# 	•	“...”
# ...
# 2) Weakly Risky Terms Reasons:

# 	•	“..."
# ...
# 	•	“...”
# ...

# 1) Risky Terms Suggested Changes:
# 	•	“A non-renewal notice must be sent the following amount of time prior to expiration.”
# Revise to: “A non-renewal notice must be sent at least 30 days prior to the expiration date.”
# 	•	“...”
# Revise to: "..."
# 2) Weakly Risky Terms Suggested Changes:
# 	•	“...”
# Revise to: "..."
# 	•	“...”
# Revise to: "..."

# Now analyze the following contract:
# """

# sys_template = template = """
# You are 'Secure Your Contract', an AI-based assistant for drafting contracts that takes a contract as an input and provides an anlalysis following these steps:

# ### Steps:
# 1. Detects risky and weakly risky terms (if existing).
# 2. Provide the reasons for detecting them as risky or weakly risky ones.
# 3. Provide refinement suggestions towards the contract without possible disadvantage.

# Also there are some guidlines to follow:

# ### Guideline about analysis:
# 1. Directly refer to the parts of problematic terms with \"\" rather than abstract or shortened representation (e.g. ~ cluases, ...) of them.
# 2. The number of risky/weakly risky terms depends on the content.
# 3. Avoid redundant descriptions or output.

# ### Guideline about format:
# 1. For step 1. (detection), only provide detected terms without additional comments.
# 2. For step 2. (reasons), provide detected the terms and the reasons.
# 3. For step 3. (suggesion), only provide detected terms and corresponding refinement starting with 'Revise to:'.

# Now analyze the following contract:
# """


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
def remove_stop_words(text):
    return ' '.join([word for word in text.split() if word.lower() not in stop_words])


config = OmegaConf.load("../configs/data2hf.yml")
with open(config.template_path, "r") as file:
    sys_template = file.read() 


sys_template = remove_stop_words(sys_template)


def prompt_with_template(prompt):

    # stopwords removal

    prompt = remove_stop_words(prompt)

    return [
    {
        "role": "system",
       #"content": "You are \'Secure Your Contract\' that takes a contract as an input and 1. detects risky and weakly risky terms/phrases (if existing) 2. provide the reasons for them 3. provide refinement suggestion towards the contract without possible disadvantage. Note 1. directly refer to the parts of problematic terms with \"\" rather than abstract representation (e.g. ~ cluases, ...) of them, 2. the number of risky/weakly risky terms depends on the content.",
        "content": f"{sys_template}"
    },
    # {"role": "user", "content": "\n### Input Contract:\n" + prompt + "\n### Risky/Weakly Risky Terms Analysis:\n"},
    {"role": "user", "content": "\n### Input Contract:\n" + prompt},
]

def set_all_seeds(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    set_seed(seed)