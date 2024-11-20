import os
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load risky terms
risky_terms = [
    "compulsory", "mandatory", "mutual agreement", "termination", "indefinite",
    "confidentiality", "probation", "assignment", "discretion", "voidable",
    "irrevocable", "reasonable", "in good faith", "best efforts", "at the discretion of",
    "as determination by", "reasonable notice", "subject to approval", "customary",
    "time is of the essence", "upon mutual consent", "for convenience", "unless otherwise agreed"
]

# Paths to contract and GT folders
contract_folder = "./data/test/contract"
gt_folder = "./data/test/GT"

# Helper function: Load all files from a folder
def load_texts_from_folder(folder_path):
    texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r") as file:
                texts.append(file.read())
    return texts

# Load texts
contract_texts = load_texts_from_folder(contract_folder)
gt_texts = load_texts_from_folder(gt_folder)

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function: Tokenize, remove stopwords, and lemmatize words
def tokenize_and_process(texts):
    words = []
    for text in texts:
        # Tokenize and process
        tokens = re.findall(r'\b\w+\b', text.lower())  # Tokenize into words
        processed_words = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]
        words.extend(processed_words)
    return Counter(words)

# Function: Extract risky terms from text (with stopwords removal and lemmatization)
def extract_risky_terms(texts, terms):
    found_terms = []
    for text in texts:
        # Tokenize and process
        tokens = re.findall(r'\b\w+\b', text.lower())
        processed_tokens = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]

        # processed_tokens = processed_tokens[:10]

        # print(processed_tokens)
        for term in terms:
            # print(term)
            # print(' '.join(processed_tokens))
            # if any(re.search(rf'\b{re.escape(term)}\b', ' '.join(processed_tokens), re.IGNORECASE)):
                # found_terms.append(term)
            match = re.search(rf'\b{re.escape(term)}\b', ' '.join(processed_tokens), re.IGNORECASE)
            if match:  # This will check if a match object is found (i.e., not None)
                found_terms.append(term)
    return Counter(found_terms)

# a. Word Distribution in Training Set (after stopword removal and lemmatization)
contract_word_counts = tokenize_and_process(contract_texts)
wordcloud_contract = WordCloud(width=800, height=400).generate_from_frequencies(contract_word_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_contract, interpolation="bilinear")
plt.axis("off")
plt.title("Word Distribution in Training Set (Contracts)")
plt.show()
plt.savefig("./a.png")

# b. Word Distribution (Filtered by Risky Terms)

risky_term_counts = extract_risky_terms(contract_texts, risky_terms) # contract_texts : a list of each contract
wordcloud_risky = WordCloud(width=800, height=400).generate_from_frequencies(risky_term_counts)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_risky, interpolation="bilinear")
plt.axis("off")
plt.title("Word Distribution (Filtered by Risky Terms)")
plt.show()
plt.savefig("./b.png")

# c. Unfiltered Word Distribution (TF-IDF after stopword removal and lemmatization)
tfidf_vectorizer = TfidfVectorizer(stop_words="english")  # Automatically remove stopwords in TF-IDF
tfidf_matrix = tfidf_vectorizer.fit_transform(contract_texts)
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
tfidf_scores = dict(zip(tfidf_feature_names, tfidf_matrix.sum(axis=0).A1))
tfidf_sorted = dict(sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:50])

wordcloud_tfidf = WordCloud(width=800, height=400).generate_from_frequencies(tfidf_sorted)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_tfidf, interpolation="bilinear")
plt.axis("off")
plt.title("Unfiltered Word Distribution (TF-IDF)")
plt.show()
plt.savefig("./c.png")

# e. Compare Training Set and GT (Hallucination Check)
gt_word_counts = tokenize_and_process(gt_texts)
missing_terms = {term: gt_word_counts[term] for term in gt_word_counts if term not in contract_word_counts}

plt.bar(missing_terms.keys(), missing_terms.values())
plt.xticks(rotation=90)
plt.title("Terms in GT Missing from Training Set (Hallucination Check)")
plt.show()
plt.savefig("./d.png")