import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalpha()]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    # Calculate TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([' '.join(lemmatized_tokens)])
    feature_names = vectorizer.get_feature_names_out()
    dense = tfidf_matrix.todense()
    denselist = dense.tolist()
    tfidf_scores = denselist[0]
    
    # Create a dictionary of keywords and their scores
    keywords = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}
    
    # Filter keywords based on a threshold
    filtered_keywords = {word: score for word, score in keywords.items() if score > 0.1}
    
    return filtered_keywords