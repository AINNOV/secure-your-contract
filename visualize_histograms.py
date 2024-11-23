import os
import numpy as np
import re
import json
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer
from matplotlib.colors import LinearSegmentedColormap


contract_folder = "./data/test/contract"
gt_folder = "./data/test/GT"


def load_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as file:
                text = file.read()
                if text.strip() == '': continue
                texts.append(text)
    return texts


#############################################################################
# e. Histogram of Token Counts for Generated Train Contracts w/ Prompting
#############################################################################

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
with open('./data/raw/SYC_train_with_testPDF.json', 'r') as f:
    data = json.load(f)

total_contracts = 100
token_counts = []

for i in range(total_contracts):
    contract = data[i % len(data)] 
    contract_text = contract['prompt'] 
    tokens = tokenizer.encode(contract_text, truncation=True, padding=False)
    cnt = len(tokens)
    token_counts.append(cnt)

counts, bins = np.histogram(token_counts, bins=50)
cmap = LinearSegmentedColormap.from_list("earthy_gradient", ["#8E735B", "#C3B091", "#F2E2A5", "#C9D98A", "#7F9A4E"])

norm = plt.Normalize(counts.min(), counts.max())
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(counts)):
    ax.bar(bins[i], counts[i], width=bins[i+1] - bins[i], 
           color=cmap(norm(counts[i])), edgecolor="black")

ax.set_title("Histogram of Token Counts for Generated Train Contracts w/ Prompting", fontsize=16)
ax.set_xlabel("Number of Tokens", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig("./e_gradation.png")


#############################################################################
# f. Histogram of Token Counts for Scraped Train Contracts (WONDER.LEGAL)
#############################################################################

real_contract_folder = "./data/pair/prompt"
data = load_texts_from_folder(real_contract_folder)

total_contracts = 100
token_counts = []

for i in range(total_contracts):
    contract = data[i % len(data)] 
    tokens = tokenizer.encode(contract, truncation=True, padding=False)
    cnt = len(tokens)
    token_counts.append(cnt)

counts, bins = np.histogram(token_counts, bins=50)
cmap = LinearSegmentedColormap.from_list("earthy_gradient", ["#8E735B", "#C3B091", "#F2E2A5", "#C9D98A", "#7F9A4E"])

norm = plt.Normalize(min(counts), max(counts))
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(counts)):
    ax.bar(bins[i], counts[i], width=bins[i + 1] - bins[i], 
            color=cmap(norm(counts[i])), edgecolor="black")

ax.set_title("Histogram of Token Counts for Scraped Train Contracts (WONDER.LEGAL)", fontsize=16)
ax.set_xlabel("Number of Tokens", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.show()
plt.savefig("./f_gradation.png")


#############################################################################
# e. Histogram of Token Counts for Test Contracts
#############################################################################


real_contract_folder = "./data/test/contract"
data = load_texts_from_folder(real_contract_folder)

total_contracts = 10
token_counts = []

for i in range(total_contracts):
    contract = data[i] 
    tokens = tokenizer.encode(contract, truncation=True, padding=False)
    cnt = len(tokens)
    token_counts.append(cnt)

counts, bins = np.histogram(token_counts, bins=50)
cmap = LinearSegmentedColormap.from_list("earthy_gradient", ["#8E735B", "#C3B091", "#F2E2A5", "#C9D98A", "#7F9A4E"])

norm = plt.Normalize(min(counts), max(counts))
fig, ax = plt.subplots(figsize=(10, 6))

for i in range(len(counts)):
    ax.bar(bins[i], counts[i], width=bins[i + 1] - bins[i], 
            color=cmap(norm(counts[i])), edgecolor="black")

ax.set_title("Histogram of Token Counts for Test Contracts", fontsize=16)
ax.set_xlabel("Number of Tokens", fontsize=12)
ax.set_ylabel("Frequency", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()
plt.savefig("./g_gradation.png")