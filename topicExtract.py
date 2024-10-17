from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from gensim.models import LdaModel

def extract_topics_from_summary(summary, num_topics=1, num_words=3):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(summary)
    
    filtered_tokens = [
        lemmatizer.lemmatize(word.lower())
        for word in tokens
        if word.isalpha() and word.lower() not in stop_words
    ]

    dictionary = corpora.Dictionary([filtered_tokens])
    corpus = [dictionary.doc2bow(filtered_tokens)]

    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

    topics = lda_model.print_topics(num_words=num_words)
    
    extracted_topics = []
    for topic in topics:
        extracted_topics.append(topic[1])  

    return extracted_topics