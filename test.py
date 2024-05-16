import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

from spellchecker import SpellChecker

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

import requests as rq

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from transformers import BertTokenizer, BertModel
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv('data/rugs-bert-description.csv')
bert_corpus = df.values

df = pd.read_csv('data/rugs-stemmed-description.csv')
stemmer_corpus = df['combined'].tolist()
df = pd.read_csv('data/rugs-lemmatized-description.csv')
lemmatizer_corpus = df['combined'].tolist()

countvectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()

spell = SpellChecker()

def tokenize_bert(text):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Tokenize and correct the text
    tokenized_text = word_tokenize(text)
    corrected_text = [spell.correction(word) for word in tokenized_text]
    text = ' '.join(corrected_text)

    # Tokenize and encode the search query
    encoded = bert_tokenizer.encode(text, add_special_tokens=True)
    tokenized = torch.tensor(encoded).unsqueeze(0).to(device)
    
    # Get BERT embeddings
    with torch.no_grad():
        embeddings = model_bert(tokenized)[0][0].mean(dim=0)

    return embeddings.cpu().numpy()

def tokenizer_lemmatizer(text):
    # Tokenize the description
    tokenized_text = word_tokenize(text)
    # Remove english stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_text = [word for word in tokenized_text if word not in stop_words]
    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    corrected_text = [lemmatizer.lemmatize(spell.correction(word)) for word in cleaned_text]

    text = ' '.join(corrected_text)
    return text

def tokenizer_stemmer(text):
    # Tokenize the description
    tokenized_text = word_tokenize(text)
    # Remove english stopwords
    stop_words = set(stopwords.words('english'))
    cleaned_text = [word for word in tokenized_text if word not in stop_words]
    # Perform stemming
    stemmer = PorterStemmer()
    corrected_text = [stemmer.stem(spell.correction(word)) for word in cleaned_text]

    text = ' '.join(corrected_text)
    return text

def euclidean_similar_rugs(search_query, tokenizer='lemmatizer', top_n=10):
    if tokenizer.lower()=='stemmer':
        search_query = tokenizer_stemmer(search_query)
        count_matrix = countvectorizer.fit_transform(stemmer_corpus)

    elif tokenizer.lower()=='lemmatizer':
        search_query = tokenizer_lemmatizer(search_query)
        count_matrix = countvectorizer.fit_transform(lemmatizer_corpus)
    else:
        print('tokenizer not found!!')
        return 404
    print(search_query)

    query_vector = countvectorizer.transform([search_query])

    euclidean_dist = euclidean_distances(query_vector, count_matrix)
    
    sort_euclidean_dist = euclidean_dist.flatten()
    sorted_indices_asc = sort_euclidean_dist.argsort()

    similar_indices_asc = sorted_indices_asc[:top_n]
    
    return df.iloc[similar_indices_asc]

def cosine_similar_rugs(search_query, tokenize='stemmer', top_n=10):
    if tokenize.lower()=='stemmer':
        search_query = tokenizer_stemmer(search_query)
        vector = vectorizer.fit_transform(stemmer_corpus)
        query_vector = vectorizer.transform([search_query])
    elif tokenize.lower()=='lemmatizer':
        search_query = tokenizer_lemmatizer(search_query)
        vector = vectorizer.fit_transform(lemmatizer_corpus)
        query_vector = vectorizer.transform([search_query])
    elif tokenize.lower()=='bert':
        embeddings = tokenize_bert(search_query)
        vector = df.values
        query_vector = embeddings.reshape(1, -1)
    else:
        print('tokenizer not found!!')
        return 404
    
    cosine_sim = cosine_similarity(query_vector, vector)
    
    sort_cosine_sim = cosine_sim.flatten()
    sorted_indices_desc = sort_cosine_sim.argsort()[::-1]

    similar_indices_desc = sorted_indices_desc[:top_n]
    
    return df.iloc[similar_indices_desc]

def knn_similar_rugs(search_query, tokenizer='lemmatizer', top_n=10):
    if tokenizer.lower()=='stemmer':
        search_query = tokenizer_stemmer(search_query)
        tfidf_matrix = vectorizer.fit_transform(stemmer_corpus)

    elif tokenizer.lower()=='lemmatizer':
        search_query = tokenizer_lemmatizer(search_query)
        tfidf_matrix = vectorizer.fit_transform(lemmatizer_corpus)
    else:
        print('tokenizer not found!!')
        return 404
    print(search_query)
    
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(tfidf_matrix)

    query_vector = vectorizer.transform([search_query])
    
    distances, indices = model_knn.kneighbors(query_vector, n_neighbors=top_n)
    
    similar_indices_desc = indices.flatten()
    
    return df.iloc[similar_indices_desc]


def plot(similar_data, data=None):
    if data == None:
        with open('data/rugs.json', 'r') as f:
            data = json.load(f)

    fig = plt.figure(figsize=(20, 10))
    for i, (index, row) in enumerate(similar_data.iterrows()):
        url = data[index]['image']
        print('\n', data[index]['description'], '\n', data[index]['materials'], '\n', url)

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = rq.get(url, headers=headers)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            
            ax = fig.add_subplot(2, 5, i+1)
            ax.imshow(image)
            ax.axis('off')
    plt.show()

# Enter the search text
# search = str(input())
search = "navy rugs"


with open('data/rugs.json', 'r') as f:
    data = json.load(f)

    print("Most similar rugs based on cosine for your search:")
    cosine_similar = cosine_similar_rugs(search, tokenize='stemmer', top_n=20)
    plot(cosine_similar[:10], data)
    plot(cosine_similar[10:], data)

    print("Most similar rugs based on euclidean for your search:")
    euclidean_similar = euclidean_similar_rugs(search, top_n=10)
    plot(euclidean_similar, data)
    
    print("Most similar rugs based on knn for your search:")
    knn_similar = knn_similar_rugs(search, top_n=10)
    plot(knn_similar, data)