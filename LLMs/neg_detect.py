## This code is designed to be a module in visualize_histograms/wordclouds.py. ##

import re
import nltk
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from nltk.chunk import ne_chunk, tree2conlltags
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gensim.summarization import keywords as gensim_keywords
from spacy.tokens import Span
import gensim
from gensim.models import FastText
from gensim import corpora
from nltk.corpus import wordnet
import fasttext.util

from omegaconf import OmegaConf
import argparse
import logging

logging.getLogger("nltk").setLevel(logging.WARNING)

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

## predefine (weakly) risky terms obtained by multiple prompting to ChatGPT, Gemini, etc. ##
risky_terms = ["compulsory", "mandatory", "mutual agreement", "termination", "indefinite", "confidentiality", "probation", "assignment", "discretion", "voidable", "irrevocable"] + ["reasonable", "in good faith", "best efforts", "at the discretion of", "as determination by", "reasonable notice", "subject to approval", "customary", "time is of the essence", "upon mutual consent", "for convenience", "unless otherwise agreed"]

risky_terms2 = [
    "compulsory", "mandatory", "mutual agreement", "termination", "indefinite",
    "confidentiality", "probation", "assignment", "discretion", "voidable",
    "irrevocable", "reasonable", "in good faith", "best efforts", "at the discretion of",
    "as determination by", "reasonable notice", "subject to approval", "customary",
    "time is of the essence", "upon mutual consent", "for convenience", "unless otherwise agreed"
]

risky_terms = list(set(risky_terms + risky_terms2))

## 1. Regex-based ##
def regex_search(contract_text):
    risky_found = []
    for term in risky_terms: # rigid -> not that effective
        if re.search(r'\b' + re.escape(term) + r'\b', contract_text, re.IGNORECASE): # detect risky terms enclosed with whitespaces, rather than split each then retrieve
            risky_found.append(term)
    return {"method": "Regex Search", "terms_found": risky_found}

## 2.SpaCy-based ##
def nltk_spacy_search(contract_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(contract_text) # tokenized by spacy -> not that effictive
    risky_found = [token.text for token in doc if token.lemma_ in risky_terms] # if lemma in risky_terms
    return {"method": "SpaCy Search", "terms_found": list(set(risky_found))}

# 3. FastText Cosine similarity-based
# def embedding_search(contract_text):

#     fasttext.util.download_model('en', if_exists='ignore')  # English
#     model = fasttext.load_model('cc.en.300.bin') # fastext word embedding

#     contract_words = word_tokenize(contract_text)
    
#     risky_found = []
#     for word in contract_words:
#         word_embedding = model.get_word_vector(word)  
        
#         for term in risky_terms:
#             term_embedding = model.get_word_vector(term) 
#             similarity = cosine_similarity([word_embedding], [term_embedding])[0][0]
            
#             if similarity > 0.7:  
#                 risky_found.append(word)
#                 break  

#     return {"method": "FastText Embedding Search", "terms_found": list(set(risky_found))}

## 3. Embedding Model Cosine similarity-based (better) ##
def embedding_search(contract_text):

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    contract_words = word_tokenize(contract_text)
    
    risky_found = []
    risky_term_embeddings = {term: model.encode([term])[0] for term in risky_terms}
    
    for word in contract_words:
        word_embedding = model.encode([word])[0]
        
        for _, term_embedding in risky_term_embeddings.items(): # detect risky terms in doc, not in pre-defined set -> very effective
            similarity = cosine_similarity([word_embedding], [term_embedding])[0][0]

            if similarity > 0.7: # threshold is heuristically set
                risky_found.append(word)
                break  
    
    return {"method": "Embedding Search", "terms_found": list(set(risky_found))}


##  4. Shallow Parsing-based ## 
def shallow_parsing_search(contract_text):

    tokens = word_tokenize(contract_text)
    tagged = pos_tag(tokens)
    
    chunks = ne_chunk(tagged)
    risky_found = []

    for word, pos, chunk in tree2conlltags(chunks): # based on NER tree -> not that effective actually
        if word.lower() in risky_terms:
            risky_found.append(word) #{"word": word, "pos": pos, "chunk": chunk}) for simplicity
    
    return {
        "method": "Shallow Parsing Search",
        "terms_found": list(set(risky_found))
    }


##  5. Sentiment ㅁnalysis-based ## 
def sentiment_analysis_search(contract_text):
    analyzer = SentimentIntensityAnalyzer()
    risky_found = []
    
    for word in contract_text.split(): # simple split
        sentiment = analyzer.polarity_scores(word)
        if sentiment['compound'] < -0.5 and word.lower() in risky_terms: # for relatively negative sentiment -> effective and unique
            risky_found.append({"word": word, "compound_score": sentiment['compound']})
            
    return {"method": "Sentiment Analysis", "terms_found": list(set(risky_found))}

##  6. TF-IDF-based ## 
def get_synonyms(word): # helper function
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower()) # synonyms
    return synonyms

def tfidf_search_with_synonyms(contract_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([contract_text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense().tolist()
    tfidf_dict = {feature_names[i]: dense[0][i] for i in range(len(feature_names))}

    expanded_risky_terms = set(risky_terms)
    for term in risky_terms:
        expanded_risky_terms.update(get_synonyms(term))
    
    risky_found = [term for term in expanded_risky_terms if tfidf_dict.get(term, 0) > 0.2] # only if the term is quite important (0.2) -> actually no use since we aggregate all and operate ,,set''
    return {"method": "TF-IDF Search", "terms_found": list(set(risky_found))}

##  7. TextRank-based ## 
def textrank_search(contract_text):
    risky_found = []
    try:
        keywords = gensim_keywords(contract_text).split('\n') # keywords extraction using TextRank -> unique
        risky_found = [term for term in keywords if term.lower() in risky_terms]
    except ValueError:
        pass 
    return {"method": "TextRank Keyword Extraction", "terms_found": list(set(risky_found))}

##  8. Dependency Parsing-based ## 
def dependency_parsing_search(contract_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(contract_text)
    risky_found = []

    for token in doc:
        if token.lemma_ in risky_terms and (token.dep_ == "nsubj" or token.dep_ == "dobj"): # only when ,,related'' to subjects or objects -> effective for contextual detection
            risky_found.append(token.text)
            
    return {"method": "Dependency Parsing", "terms_found": list(set(risky_found))}

##  9. LDA topic modeling-based ## 
def lda_search(contract_text):

    contract_words = [contract_text.split()]
    dictionary = corpora.Dictionary(contract_words)
    corpus = [dictionary.doc2bow(text) for text in contract_words]
    lda_model = gensim.models.LdaModel(corpus, num_topics = 1, id2word = dictionary, passes = 15)
    topics = lda_model.show_topic(0, topn = 10)
    risky_found = [word for word, prob in topics if word in risky_terms] # detect even if it is not in contract! -> effiective for out-of-distribution words

    return {"method": "LDA Topic Modeling", "terms_found": list(set(risky_found))}


####################################################################################################################################
####################################################################################################################################


def find_risky_terms_extended(contract_text):

    results = []
    results.append(regex_search(contract_text))
    results.append(nltk_spacy_search(contract_text))
    results.append(embedding_search(contract_text))
    results.append(shallow_parsing_search(contract_text))
    results.append(sentiment_analysis_search(contract_text))
    results.append(tfidf_search_with_synonyms(contract_text))
    results.append(textrank_search(contract_text))
    results.append(dependency_parsing_search(contract_text))
    results.append(lda_search(contract_text))
    
    return results


## Negative Keyword Detector(NKD) ##
def process_contract(text):

    risky_terms_results_extended = find_risky_terms_extended(text)
    
    output = []
    for result in risky_terms_results_extended:
        output.extend(result['terms_found'])
    
    return output # count matters for word clouds

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "inference config?")
    parser.add_argument('--inference_config', type = str, default = "../configs/inference.yml")

    args = parser.parse_args()
    config = OmegaConf.load(args.inference_config)

    with open(config.input_contract) as file:
        contract = file.read()
    
    print(f"🚨Negative Keyword Detector🚨 result over {config.input_contract.split('/')[-1] }:\n\n" + ', '.join(list(set(process_contract(contract)))) )