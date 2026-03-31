import numpy as np
import pandas as pd
import re
import string 
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

#load model
with open('static/model/model.pickle','rb') as file:
    model = pickle.load(file)

#load stopwords    
with open('static/model/corpora/stopwords/english','r') as file:
    sw = file.read().splitlines()

#load vocabulary/token   
vocab = pd.read_csv('static/model/vocabulary.txt', header=None)
token = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text): 
    data = pd.DataFrame([text], columns=['tweet'])
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', word, flags=re.MULTILINE) for word in x.split()))
    data["tweet"] = data["tweet"].apply(remove_punctuations)
    data["tweet"] = data["tweet"].str.replace('\d+', '', regex=True)
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["tweet"]

def vectorizer(ds):
    vectorized_list = []
    for sentence in ds:
        sentence_list = np.zeros(len(token))
        for i in range(len(token)):
            if token[i] in sentence.split():
                sentence_list[i] = 1

        vectorized_list.append(sentence_list)
    vectorized_lst_new = np.asarray(vectorized_list, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negetive'
    else:
        return 'possitive'
    
