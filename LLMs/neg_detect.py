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
from gensim import corpora
from omegaconf import OmegaConf

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')

risky_terms = ["compulsory", "mandatory", "mutual agreement", "termination", "indefinite", "confidentiality", "probation", "assignment", "discretion", "voidable", "irrevocable"]
weakly_risky_terms = ["reasonable", "in good faith", "best efforts", "at the discretion of", "as determination by", "reasonable notice", "subject to approval", "customary", "time is of the essence", "upon mutual consent", "for convenience", "unless otherwise agreed"]

risky_terms = risky_terms + weakly_risky_terms

# 1. Regex based
def regex_search(contract_text):
    risky_found = []
    for term in risky_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', contract_text, re.IGNORECASE):
            risky_found.append(term)
    return {"method": "Regex Search", "terms_found": risky_found}

# 2. NLTK & SpaCy based search
def nltk_spacy_search(contract_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(contract_text)
    risky_found = [token.text for token in doc if token.lemma_ in risky_terms]
    return {"method": "NLTK & SpaCy Search", "terms_found": list(set(risky_found))}

# 3. Cosine word embedding based
def embedding_search(contract_text):

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    contract_words = word_tokenize(contract_text)
    
    risky_found = []
    risky_term_embeddings = {term: model.encode([term])[0] for term in risky_terms}
    
    for word in contract_words:
        word_embedding = model.encode([word])[0]
        
        for term, term_embedding in risky_term_embeddings.items():
            similarity = cosine_similarity([word_embedding], [term_embedding])[0][0]

            if similarity > 0.7: 
                risky_found.append(word)
                break  
    
    return {"method": "Embedding Search", "terms_found": list(set(risky_found))}


# 5. Shallow Parsing bases
def shallow_parsing_search(contract_text):
    tokens = word_tokenize(contract_text)
    tagged = pos_tag(tokens)
    chunks = ne_chunk(tagged)
    risky_found = []
    
    for i in tree2conlltags(chunks):
        word, pos, chunk = i
        if word.lower() in risky_terms and chunk != 'O':
            risky_found.append(word)
    
    return {"method": "Shallow Parsing Search", "terms_found": list(set(risky_found))}



# 6. Sentiment analysis based
def sentiment_analysis_search(contract_text):
    analyzer = SentimentIntensityAnalyzer()
    risky_found = []
    
    for word in contract_text.split():
        sentiment = analyzer.polarity_scores(word)
        if sentiment['compound'] < -0.5 and word.lower() in risky_terms:
            risky_found.append(word)
            
    return {"method": "Sentiment Analysis", "terms_found": list(set(risky_found))}

# 7. TF-IDF based
def tfidf_search(contract_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([contract_text])
    feature_names = vectorizer.get_feature_names_out()
    dense = vectors.todense().tolist()
    tfidf_dict = {feature_names[i]: dense[0][i] for i in range(len(feature_names))}
    
    risky_found = [term for term in risky_terms if tfidf_dict.get(term, 0) > 0.1]
    return {"method": "TF-IDF Search", "terms_found": list(set(risky_found))}

# 8. TextRank & keyword extraction based
def textrank_search(contract_text):
    risky_found = []
    try:
        keywords = gensim_keywords(contract_text).split('\n')
        risky_found = [term for term in keywords if term.lower() in risky_terms]
    except ValueError:
        pass  # keywords 함수에 문제가 있을 경우 넘어감
    return {"method": "TextRank Keyword Extraction", "terms_found": list(set(risky_found))}

# 9. Dependency parsing based
def dependency_parsing_search(contract_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(contract_text)
    risky_found = []

    for token in doc:
        if token.lemma_ in risky_terms and (token.dep_ == "nsubj" or token.dep_ == "dobj"):
            risky_found.append(token.text)
            
    return {"method": "Dependency Parsing", "terms_found": list(set(risky_found))}

# 10. LDA topic modeling based
def lda_search(contract_text):
    contract_words = [contract_text.split()]
    dictionary = corpora.Dictionary(contract_words)
    corpus = [dictionary.doc2bow(text) for text in contract_words]
    lda_model = gensim.models.LdaModel(corpus, num_topics=1, id2word=dictionary, passes=15)
    topics = lda_model.show_topic(0, topn=10)
    risky_found = [word for word, prob in topics if word in risky_terms]

    return {"method": "LDA Topic Modeling", "terms_found": list(set(risky_found))}


def find_risky_terms_extended(contract_text):
    results = []
    results.append(regex_search(contract_text))
    results.append(nltk_spacy_search(contract_text))
    results.append(embedding_search(contract_text))
    results.append(shallow_parsing_search(contract_text))
    results.append(sentiment_analysis_search(contract_text))
    results.append(tfidf_search(contract_text))
    results.append(textrank_search(contract_text))
    results.append(dependency_parsing_search(contract_text))
    results.append(lda_search(contract_text))
    
    return results

config = OmegaConf.load("../configs/inference.yml")
with open(config.input_contract, "r") as file:
    contract_text = file.read()


risky_terms_results_extended = find_risky_terms_extended(contract_text)


for result in risky_terms_results_extended:
    print(result)