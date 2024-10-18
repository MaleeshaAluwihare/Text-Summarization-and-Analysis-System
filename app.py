import streamlit as st
import os
import torch
import time
from transformers import BartTokenizer,BertTokenizer, BartForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
from Keyword_extraction import extract_keywords
from topicExtract import extract_topics_from_summary
from pymongo import MongoClient
import bcrypt
import uuid
import PyPDF2 
import docx2txt
from datetime import datetime
from streamlit_lottie import st_lottie
import json
from dotenv import load_dotenv

load_dotenv()

#dialog model directory
bart_samsum = 'BART_Finetuned'
bart_medical = 'BART_Finetuned_Medical'
sentiment = 'Sentimental_Bestmodel'

# Cache the loading of models to reduce time
@st.cache_resource
def load_dialog_model():
    return BartForConditionalGeneration.from_pretrained(bart_samsum)

@st.cache_resource
def load_health_model():
    return BartForConditionalGeneration.from_pretrained(bart_medical)

@st.cache_resource
def load_legal_model():
    return BartForConditionalGeneration.from_pretrained(bart_samsum)

@st.cache_resource
def load_sentiment_model():
    model = torch.load("Sentimental_Bestmodel/Sentiment_model.pth", map_location=torch.device('cpu'))
    return model

@st.cache_resource
def load_topic_model():
    return AutoModelForSequenceClassification.from_pretrained('BART_Finetuned')

@st.cache_resource
def load_tokenizers():
    bart_tokenizer = BartTokenizer.from_pretrained('BART_Finetuned')
    sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    topic_tokenizer = AutoTokenizer.from_pretrained('BART_Finetuned')
    return bart_tokenizer, sentiment_tokenizer, topic_tokenizer

# Load models and tokenizers
dialog_model = load_dialog_model()
health_model = load_health_model()
legal_model = load_legal_model()
sentiment_model = load_sentiment_model()
topic_model = load_topic_model()
bart_tokenizer, sentiment_tokenizer, topic_tokenizer = load_tokenizers()

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

# Function to load Lottie animation from a file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
#MongoDBConnection
mongodb_uri = os.getenv("MONGODB_URI")
client = MongoClient(mongodb_uri)
db = client.TextSummarization
users_collection = db.Users
history_collection = db.UserHistory

#create new user
def create_user(username, password):
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user_id = str(uuid.uuid4()) 
    users_collection.insert_one({"username": username, "password": hashed_password, "user_id": user_id})

#check if user exists for session
def check_user(username, password):
    user = users_collection.find_one({"username": username})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return user['user_id']
    return None

#store user history
def log_user_history(user_id, text, summary):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history_collection.insert_one({"user_id": user_id, "text": text, "summary": summary,"created_at": current_time})


# Display user history
def display_user_history(user_id):
    st.subheader("Your Summarization History")
    history = list(history_collection.find({"user_id": user_id}).sort("created_at", -1))
    
    if len(history) == 0:
        st.info("It looks like you haven't summarized any text yet. Start by entering some text and creating your first summary!")
    else:
        for record in history:
            st.markdown(f"**Date**: {record['created_at']}")
            with st.expander(f"**Summary**: {record['summary']}"):
                st.markdown(f"**Text**: {record['text']}")


# check if username already exists
def check_user_exists(username):
    if users_collection.find_one({"username": username}):
        return True
    return False

def login():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://raw.githubusercontent.com/MaleeshaAluwihare/Text-Summarization-and-Analysis-System/main/Image/ai-text-summarizer.png");
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

def register():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://raw.githubusercontent.com/MaleeshaAluwihare/Text-Summarization-and-Analysis-System/main/Image/ai-text-summarizer.png");
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("Create Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    
    if st.button("Create Account"):
        if check_user_exists(username):
            st.error("Username already exists! Please choose a different username.")
        else:
            create_user(username, password)
            st.success(f"Account created successfully!")

def summarization_page():
    summarization_page()

def main():
    st.sidebar.title("📋 Welcome to Summarizer!")
    st.sidebar.write("Choose an option to continue:")
    
    choice = st.sidebar.radio(
        "Go to",
        ["🔑 Login", "🙎 Register"]
    )
    st.sidebar.markdown("---") 

    if 'user_id' in st.session_state:
        st.sidebar.success(f"Welcome, {st.session_state['username']}!")
        if st.sidebar.button("🔓 Log Out"):
            del st.session_state['user_id']
            del st.session_state['username']
            st.rerun()
        summrization_page()
    else:
        if choice == "🙎 Register":
            register()
        elif choice == "🔑 Login":
            login()

# Document read
def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    return docx2txt.process(file)

def summrization_page():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #000000;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title(f"Welcome {st.session_state.username} 👋")
    st.markdown('---')
    st.title("Text Summarization and Analysis")

    # Select paragraph type
    paragraph_type = st.selectbox("Select Paragraph Type📄", ("Dialog", "Health", "Legal", "Artical"))

    input_method = None

    # If text is entered, disable file upload
    paragraph = st.text_area(label="Enter the paragraph📝", disabled=st.session_state.get('file_uploaded', False))

    uploaded_file = st.file_uploader("Upload a document (TXT, PDF, DOCX)", type=["txt", "pdf", "docx"], disabled=st.session_state.get('text_entered', False))


  # Handle text input action
    if paragraph and not st.session_state.get('text_entered', False):
        st.session_state['text_entered'] = True
        st.session_state['file_uploaded'] = False

    # Handle file upload action
    if uploaded_file and not st.session_state.get('file_uploaded', False):
        st.session_state['file_uploaded'] = True
        st.session_state['text_entered'] = False
        if uploaded_file.type == "text/plain":
            text = read_txt(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            text = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            text = read_docx(uploaded_file)
        else:
            st.error("Unsupported file format")
        paragraph = text


    # Options for additional processing
    options = st.multiselect("Additional Options🔗", ["Topic Generation", "Sentiment Analysis", "Word Extraction"])

    # Load appropriate model based on the paragraph type
    if paragraph_type == "Dialog":
        selected_model = dialog_model
    elif paragraph_type == "Health":
        selected_model = health_model
    elif paragraph_type == "Artical":
        selected_model = dialog_model
    else:
        selected_model = legal_model

    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            paragraph = read_txt(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            paragraph = read_pdf(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            paragraph = read_docx(uploaded_file)
        else:
            st.error("Unsupported file format")

    lottie_animation = load_lottiefile(r"Animation\loading.json")

    # Summarization button
    if st.button("Summarize"):

        lottie_placeholder = st.empty()
        with lottie_placeholder.container():
            st.markdown('<div class="centered-container">', unsafe_allow_html=True)
            st_lottie(lottie_animation, height=150, width=300)
            st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(25)
        lottie_placeholder.empty()

        inputs = bart_tokenizer(paragraph, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = selected_model.generate(inputs['input_ids'], num_beams=4, max_length=100, early_stopping=True)
        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        log_user_history(st.session_state.user_id,paragraph,summary)

        st.markdown("### Summary:")
        st.success(summary)

        # Perform additional options if selected
        if "Sentiment Analysis" in options:
            st.markdown('---')
            st.markdown("### Sentiment Analysis:")
            
            sentiment, article_type = predict_sentiment_and_type(paragraph)
            sentiment_labels = ['Negative', 'Neutral', 'Positive']
            article_labels = ['Business', 'Entertainment', 'General', 'Health', 'Science', 'Sports', 'Technology']
            
            sentiment_display = f"""
                <div style='background-color:#f1f1f1; color:black; padding:8px 15px; border-radius:25px; display:inline-block; margin-right: 10px; margin-bottom: 10px;'>
                    Sentiment: {sentiment_labels[sentiment]}
                </div>
            """
            article_display = f"""
                <div style='background-color:#f1f1f1; color:black; padding:8px 15px; border-radius:25px; display:inline-block;'>
                    Article Type: {article_labels[article_type]}
                </div>
            """
            st.markdown(sentiment_display, unsafe_allow_html=True)
            st.markdown(article_display, unsafe_allow_html=True)


        if "Topic Generation" in options:
            st.markdown('---')
            st.markdown("### Generated Topics:")

            topics = extract_topics_from_summary(paragraph)

            cleaned_topics = []
            for topic in topics:
                words = [item.split('*')[1].replace('"', '').strip() for item in topic.split('+')]
                cleaned_topics.append(words) 
            
            for topic_words in cleaned_topics:
                for word in topic_words:
                    st.markdown(f"<div style='background-color:#f1f1f1; color:black; padding:8px 15px; border-radius:25px; display:inline-block; margin-bottom: 10px;'>{word}</div>", unsafe_allow_html=True)


        if "Word Extraction" in options:
            st.markdown('---')
            st.markdown("### Extracted Keywords:")

            keywords = extract_keywords(summary)
            num_keywords = len(keywords)
            columns = st.columns(min(4, num_keywords))
            
            for i, keyword in enumerate(keywords):
                with columns[i % 4]:
                    st.markdown(
                        f"<div style='background-color:#f1f1f1; color:black; padding:12px 25px; border-radius:25px; "
                        f"display:inline-block; margin-bottom: 15px;'>{keyword}</div>",
                        unsafe_allow_html=True
                    )


    st.markdown('---')
    display_user_history(st.session_state.user_id)  

if __name__ == "__main__":
    if 'text_entered' not in st.session_state:
        st.session_state['text_entered'] = False
    if 'file_uploaded' not in st.session_state:
        st.session_state['file_uploaded'] = False
    main()