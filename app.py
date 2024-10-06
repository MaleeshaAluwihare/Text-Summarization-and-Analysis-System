from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.document_loaders import PyPDFLoader,DirectoryLoader
# from langchain.chains.summarize import load_summarize_chain
from transformers import BertModel
import torch.nn as nn
import torch
import base64
import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
from Custom_Sentiment_model import BertForMultiTask
from pymongo import MongoClient
import bcrypt
import uuid

#dialog model directory
bart_samsum = 'BART_Finetuned'
sentiment = 'Sentimental_Bestmodel'

# Initialize the custom model
model = BertForMultiTask('bert-base-uncased', num_sentiment_labels=3, num_article_labels=7)

# Load models
dialog_model = BartForConditionalGeneration.from_pretrained(bart_samsum)
health_model = BartForConditionalGeneration.from_pretrained(bart_samsum)
legal_model = BartForConditionalGeneration.from_pretrained(bart_samsum)

# Create an instance of the custom model
sentiment_model = BertForMultiTask('bert-base-uncased', num_sentiment_labels=3, num_article_labels=7)
topic_model = AutoModelForSequenceClassification.from_pretrained(bart_samsum)

# Tokenizers
tokenizer = BartTokenizer.from_pretrained(bart_samsum)
sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
topic_tokenizer = AutoTokenizer.from_pretrained(bart_samsum)

# Load the trained model weights
sentiment_model = torch.load("Sentimental_Bestmodel/Sentiment_model.pth", map_location=torch.device('cpu'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_model.to(device)
sentiment_model.eval()

def predict_sentiment_and_type(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        sentiment_logits, article_logits = sentiment_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Predict sentiment and article type
        sentiment_pred = torch.argmax(sentiment_logits, dim=1).item()
        article_pred = torch.argmax(article_logits, dim=1).item()
    
    return sentiment_pred, article_pred

#MongoDBConnection
client = MongoClient("mongodb+srv://malshaconline:0216y0i6wyrScneu@summarization.gnxkc.mongodb.net/")
db = client.TextSummarization
users_collection = db.Users
history_collection = db.UserHistory

#create new user
def create_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_id = str(uuid.uuid4()) 
    users_collection.insert_one({"username": username, "password": hashed_password, "user_id": user_id})

#check if user exists
def check_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return user['user_id']
    return None

#store user history
def log_user_history(user_id, text, summary):
    history_collection.insert_one({"user_id": user_id, "text": text, "summary": summary})

#User registration
def register():
    st.title("Register")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Create Account"):
        create_user(username, password)
        st.success(f"Account created successfully!")

# User login
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Login"):
        user_id = check_user(username, password)
        if user_id:
            st.session_state.user_id = user_id
            st.session_state.username = username
            st.success("Logged in successfully!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

#display user history
def display_user_history(user_id):
    st.subheader("Your Summarization History")
    history = history_collection.find({"user_id": user_id})
    for record in history:
        st.write(f"Text: {record['text']}")
        st.write(f"Summary: {record['summary']}")
        st.markdown('---')

def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to",["Login","Register"])

    if choice == "Register":
        register()
    if 'user_id' in st.session_state:
        summrization_page()
    elif choice == "Login":
        login()


def summrization_page():
    st.title("Text Summarization and Analysis")

    # Select paragraph type
    paragraph_type = st.selectbox("Select Paragraph Type", ("Dialog", "Health", "Legal", "Artical"))

    # Input paragraph
    paragraph = st.text_area(label="Enter the paragraph")

    # Options for additional processing
    options = st.multiselect("Additional Options", ["Topic Generation", "Sentiment Analysis", "Word Extraction"])

    # Load appropriate model based on the paragraph type
    if paragraph_type == "Dialog":
        selected_model = dialog_model
    elif paragraph_type == "Health":
        selected_model = health_model
    else:
        selected_model = legal_model

    # Summarization button
    if st.button("Summarize"):
        inputs = tokenizer(paragraph, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = selected_model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        log_user_history(st.session_state.user_id,paragraph,summary)
        st.success(summary)

        display_user_history(st.session_state.user_id)

    # Perform additional options if selected
    if "Sentiment Analysis" in options:
        if st.button("Analyze Sentiment"):
            sentiment, article_type = predict_sentiment_and_type(paragraph)
            st.write(f"Sentiment: {sentiment}")
            st.write(f"Article Type: {article_type}")

    if "Topic Generation" in options:
        if st.button("Generate Topic"):
            inputs = topic_tokenizer(paragraph, return_tensors="pt")
            topic_logits = topic_model(**inputs).logits
            topic = topic_logits.argmax(dim=-1).item()
            topic_labels = {0: 'Business', 1: 'Entertainment', 2: 'General', 3: 'Health', 4: 'Science', 5: 'Sports', 6: 'Technology'}
            st.write("**Topic:**", topic_labels[topic])

if __name__ == "__main__":
    main()