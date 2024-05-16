import euclidean_model as em
import cosine_similar_model as cs
import knn_model as knn

print('\nEnter the search query:')
search = str(input('> '))
print('Enter the wanted number of similar results:')
n = int(input('> '))

print("\nEnter the index of the wanted model to use:\n1 for euclidean comparisor.\n2 for cosine similarity comparisor.\n3 for knn comparisor.\n")
comp_idx = int(input('> '))

if comp_idx == 1:
    print("\nEnter the index of the wanted vectorizer to use with NLTK tokenizer :\n1 for 'BERT' embedding.\n2 for 'Stemmer' vectorizer.\n3 for 'Lemmatizer' vectorizer.\n")
    vect_idx = int(input('> '))

    if vect_idx == 1:
        bert = em.euclidean(search, n, 'bert')
    elif vect_idx == 2:
        bert = em.euclidean(search, n, 'stemmer')
    elif vect_idx == 3:
        bert = em.euclidean(search, n, 'lemmatizer')
    else:
        print('No such vectorizer found!!')

elif comp_idx == 2:
    print("\nEnter the index of the wanted vectorizer to use with NLTK tokenizer :\n1 for 'BERT' embedding.\n2 for 'Stemmer' vectorizer.\n3 for 'Lemmatizer' vectorizer.\n")
    vect_idx = int(input('> '))

    if vect_idx == 1:
        bert = cs.cosin_sim(search, n, 'bert')
    elif vect_idx == 2:
        bert = cs.cosin_sim(search, n, 'stemmer')
    elif vect_idx == 3:
        bert = cs.cosin_sim(search, n, 'lemmatizer')
    else:
        print('No such vectorizer found!!')

elif comp_idx == 2:
    print("\nEnter the index of the wanted vectorizer to use with NLTK tokenizer :\n1 for 'BERT' embedding.\n2 for 'Stemmer' vectorizer.\n3 for 'Lemmatizer' vectorizer.\n")
    vect_idx = int(input('> '))

    if vect_idx == 1:
        bert = knn.knn(search, n, 'bert')
    elif vect_idx == 2:
        bert = knn.knn(search, n, 'stemmer')
    elif vect_idx == 3:
        bert = knn.knn(search, n, 'lemmatizer')
    else:
        print('No such vectorizer found!!')

else:
    print('No such model index found!!')