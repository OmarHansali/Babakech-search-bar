import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

import requests as rq

from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt


import tokenizes as tk

class knn():
    def __init__(self, search, top_n=20, tokenizer='lemmatizer'):
        self.plot(search, top_n, tokenizer)

    def knn_similar_rugs(self, search_query, top_n, tokenizer):
        vectorizer = TfidfVectorizer()
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')

        if tokenizer.lower() == 'stemmer':
            search_query = tk.tokenizer_stemmer(search_query)
            
            df = pd.read_csv('data/rugs-stemmed-description.csv')
            stemmer_corpus = df['combined'].tolist()

            vector = vectorizer.fit_transform(stemmer_corpus)
            query_vector = vectorizer.transform([search_query])
        elif tokenizer.lower() == 'lemmatizer':
            search_query = tk.tokenizer_lemmatizer(search_query)
            
            df = pd.read_csv('data/rugs-lemmatized-description.csv')
            lemmatizer_corpus = df['combined'].tolist()

            vector = vectorizer.fit_transform(lemmatizer_corpus)
            query_vector = vectorizer.transform([search_query])
        elif tokenizer.lower() == 'bert':
            embeddings, search_query = tk.tokenizer_bert(search_query)
            
            df = pd.read_csv('data/rugs-bert-description.csv')

            vector = df.values
            query_vector = embeddings.reshape(1, -1)

        else:
            print('tokenizer not found!!')
            return 404
    
        model_knn.fit(vector)
        
        distances, indices = model_knn.kneighbors(query_vector, n_neighbors=top_n)
        
        similar_indices_desc = indices.flatten()
        sorted_indices_asc = similar_indices_desc.argsort()

        similar_indices_asc = sorted_indices_asc[:top_n]
    
        return df.iloc[similar_indices_desc], search_query

    def plot(self, search, top_n, tokenizer):
        if tokenizer.lower() == 'bert':
            with open('data/bert-rugs.json', 'r') as f:
                data = json.load(f)
        else:
            with open('data/rugs.json', 'r') as f:
                data = json.load(f)
            
        similar_data, cor_search = self.knn_similar_rugs(search, top_n, tokenizer)
        print(f"\nMost similar rugs based on KNN for {cor_search}:")

        j = 0
        for i, (index, row) in enumerate(similar_data.iterrows()):
            if j == 0:
                fig = plt.figure(figsize=(20, 10))
            j += 1

            url = data[index]['image']
            print('\nThe index:', i+1, 'description\n', data[index]['description'], '\n', data[index]['materials'], '\n', url)

            headers = {'User-Agent': 'Mozilla/5.0'}
            response = rq.get(url, headers=headers)

            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                
                ax = fig.add_subplot(2, 5, j)
                ax.imshow(image)
                ax.axis('off')
            
            if j > 9:
                j = j-10
                plt.show()
            
            if i >= top_n-1:
                plt.show()
                break


if __name__ == '__main__':
    # Change the search text
    knn('blue carpt', 5, 'BERT')