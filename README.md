
# 📝 Text Summarization and Analysis System

A powerful system designed to take large corpora of text such as news articles, research papers, and more, and provide detailed analysis, including concise summaries, sentiment analysis, keyword extraction, and topic modeling. The system integrates state-of-the-art NLP models to streamline text processing tasks.

## 🌟 Features

- **Summarization**: Generates concise summaries from large text using the BART model, fine-tuned on multiple datasets (dialog, healthcare, legal, and news articles).
- **Sentiment Analysis**: Leverages BERT to analyze and predict the sentiment of the input text.
- **Keyword Extraction**: Identifies and extracts the most relevant keywords from the text.
- **Topic Modeling**: Implements advanced topic modeling techniques to classify and group the topics present in the text.

## 🛠️ Technology Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**: 
  - BART (Bidirectional and Auto-Regressive Transformers) for text summarization.
  - BERT (Bidirectional Encoder Representations from Transformers) for sentiment analysis and keyword extraction.
  - Topic modeling algorithms for topic identification.

## 📂 Datasets Used

The models were fine-tuned using the following datasets:

1. **Dialog Dataset**: For building models that understand conversational contexts.
2. **Healthcare Dataset**: To cater to specific summarization needs in the medical domain.
3. **Legal Dataset**: Designed to summarize and analyze legal cases and documents.
4. **Article Dataset**: Focused on news articles, research papers, and other formal texts.

## ⚙️ Installation

To run the application locally, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/MaleeshaAluwihare/text-summarization-analysis-system.git
   cd text-summarization-analysis-system
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## 🚀 Usage

### Summarization
1. Input a large corpus (e.g., research paper, news article) into the text box.
2. Click on the **Generate Summary** button to receive a summary of varying lengths.

### Sentiment Analysis
1. Input your text into the sentiment analysis section.
2. The system will return whether the text is positive, negative, or neutral.

### Keyword Extraction
1. Provide a text for keyword extraction.
2. The system will generate the most relevant keywords from the content.

### Topic Modeling
1. Enter a corpus of text to classify into specific topics.
2. The system will output the identified topics along with the relevance.

## 🤖 Models

- **BART**: Fine-tuned for summarization tasks on various datasets (Dialog, Legal, Healthcare, News Articles).
- **BERT**: Used for sentiment analysis, keyword extraction, and other NLP tasks.

## 📝 Future Improvements

- Add more fine-tuned models for domain-specific summaries (e.g., financial or academic).
- Improve keyword extraction by incorporating additional models like GPT.
- Extend the system to handle multi-lingual datasets and outputs.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
