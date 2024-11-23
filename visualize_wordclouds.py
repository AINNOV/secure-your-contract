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
import PIL.Image
import numpy as np
from LLMs.neg_detect import process_contract


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

mask_a = np.array(PIL.Image.open("./ditto_test.jpg"))
mask_b = np.array(PIL.Image.open("./pikachu_test.jpeg"))
mask_c = np.array(PIL.Image.open("./groundon.jpg"))
mask_d = np.array(PIL.Image.open("./kyogre.jpg"))


risky_terms = ["compulsory", "mandatory", "mutual agreement", "termination", "indefinite", "confidentiality", "probation", "assignment", "discretion", "voidable", "irrevocable"] + ["reasonable", "in good faith", "best efforts", "at the discretion of", "as determination by", "reasonable notice", "subject to approval", "customary", "time is of the essence", "upon mutual consent", "for convenience", "unless otherwise agreed"]

risky_terms2 = [
    "compulsory", "mandatory", "mutual agreement", "termination", "indefinite",
    "confidentiality", "probation", "assignment", "discretion", "voidable",
    "irrevocable", "reasonable", "in good faith", "best efforts", "at the discretion of",
    "as determination by", "reasonable notice", "subject to approval", "customary",
    "time is of the essence", "upon mutual consent", "for convenience", "unless otherwise agreed"
]

risky_terms = list(set(risky_terms + risky_terms2))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

contract_folder = "./data/test/GT" #"./data/test/contract"
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


contract_texts = load_texts_from_folder(contract_folder)
gt_texts = load_texts_from_folder(gt_folder)


stop_words = set(stopwords.words("english"))

lemmatizer = WordNetLemmatizer()


def tokenize_and_process(texts):
    words = []
    for text in texts:
        text = re.sub(r'<.*?>', ' ', text)  # HTML tags removal
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)# text = re.sub(r'[^\w\s]', '', text)  # special characters removal
        text = re.sub(r'\s+', ' ', text)  # spaces removal
        tokens = re.findall(r'\b\w+\b', text.lower()) # enclosed by whitespaces = token
        # tokens = tokenizer.tokenize(text)  
        processed_words = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and not re.match(r'^[\_]+$', word)
        ]
        words.extend(processed_words)
    # print(Counter(words))
    return Counter(words)

def filter_risky_terms(text):

    return process_contract(text)


#############################################################################
# a. Word Distribution in Training Set (after stopword removal and lemmatization)
#############################################################################

# contract_word_counts = tokenize_and_process(contract_texts)
# wordcloud_contract = WordCloud(width=800, height=400, mask = mask_a, background_color = "white", contour_color = "black", contour_width = 3, max_words = 50, random_state = 42).generate_from_frequencies(contract_word_counts)

# plt.figure(figsize=(8, 6))
# plt.imshow(wordcloud_contract, interpolation="bilinear")
# plt.axis("off")
# plt.title("Word Distribution in Risky Term Analyses")
# plt.show()
# plt.savefig("./a.png")


#############################################################################
############ b. Word Distribution (Filtered by Risky Terms)  ################
#############################################################################


# risky_term_counts = []
# for contract in contract_texts:
#     risky_term_counts.extend(filter_risky_terms(contract))
# wordcloud_risky = WordCloud(width=800, height=400, mask = mask_b, background_color = "white", contour_color = "black", contour_width = 3, max_words = 50, random_state = 42).generate_from_frequencies( Counter(risky_term_counts))

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud_risky, interpolation="bilinear")
# plt.axis("off")
# plt.title("Word Distribution in Risky Term Analyses\n(Filtered by Negative Keyword Extractor)")
# plt.show()
# plt.savefig("./b.png")


#############################################################################
############ c. In GT, but not in generated analyses (missing)  ################
#############################################################################

generated_folder = "./eval_output/gemini"
gt_folder = "./data/test/GT"

generated_texts = load_texts_from_folder(generated_folder)
gt_texts = load_texts_from_folder(gt_folder)

missing_counts = []
hallucination_counts = []
for generated, gt in zip(generated_texts, gt_texts):

    gt_set = set(filter_risky_terms(gt))
    generated_set = set(filter_risky_terms(generated))

    
    # missing_counts.extend(list( gt_set - generated_set ))
    hallucination_counts.extend(list( generated_set - gt_set))

# wordcloud_missing = WordCloud(width=800, height=400, mask = mask_c, background_color = "white", contour_color = "black", contour_width = 3, max_words = 50, random_state = 42).generate_from_frequencies( Counter(missing_counts))
wordcloud_hallucination = WordCloud(width=800, height=400, mask = mask_d, background_color = "white", contour_color = "black", contour_width = 3, max_words = 50, random_state = 42).generate_from_frequencies( Counter(hallucination_counts))

# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud_missing, interpolation="bilinear")
# plt.axis("off")
# plt.title("Missing\n(In GT, Not In Generated Analysis)")
# plt.show()
# plt.savefig("./c.png")

#############################################################################
############ d. In generated analyses, but not in GT (hallucination)  ################
#############################################################################

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_hallucination, interpolation="bilinear")
plt.axis("off")
plt.title("Hallucination of Gemini\n(In Generated Analysis, Not In GT)")
plt.show()
plt.savefig("./d.png")
