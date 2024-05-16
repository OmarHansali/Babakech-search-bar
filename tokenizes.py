from spellchecker import SpellChecker

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from transformers import BertTokenizer, BertModel
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spell = SpellChecker()

def tokenizer_bert(text):
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased').to(device)
    
    # Tokenize and correct the text
    tokenized_text = word_tokenize(text)
    corrected_text = [spell.correction(word) for word in tokenized_text]
    text = ' '.join(corrected_text)

    # Encode the text
    encoded = bert_tokenizer.encode(text, add_special_tokens=True)
    tokenized = torch.tensor(encoded).unsqueeze(0).to(device)
    
    # Get BERT embeddings
    with torch.no_grad():
        embeddings = model_bert(tokenized)[0][0].mean(dim=0)

    return embeddings.cpu().numpy(), text

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

if __name__ == '__main__':
    tokenizer_bert('blui rags')
    tokenizer_lemmatizer('blui rags')
    tokenizer_stemmer('blui rags')