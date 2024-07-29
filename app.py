import streamlit as st
import numpy as np
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import scipy
import gensim
from gensim.models import Word2Vec
import gensim.downloader as api
from numpy.linalg import norm
import pickle


def preprocess(text):
    # Lowercase 
    text = text.lower()
    # Tokenization
    tokens = word_tokenize(text)
    # Remove punctutaion 
    tokens = [word for word in tokens if word.isalnum()]
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens 

# Load the pretrained Word2Vec model
def load_model():  
    with open('word2vec_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model




def get_paragraph_vector(tokens,model = load_model()): 
    word_vectors = [model[word] for word in tokens if word in model]
    if not word_vectors :
        return np.zero(model.vector_size)
    paragraph_vector = np.mean(word_vectors,axis = 0)
    return paragraph_vector



# Cosine Similarity 
def cosine_sim(resume_vector,job_vector):
    cosine = np.dot(resume_vector,job_vector)/(norm(resume_vector)*norm(job_vector))
    return cosine




# Streamlit app 
st.header("Resume-Score")

job_desc = st.text_area("Enter the job descprition","")
resume_desc = st.text_area("Enter the resume descprition","")


submit = st.button("Calculate Score")


if submit:
    job_pre = preprocess(job_desc)
    resume_pre =preprocess(resume_desc)

    job_vector = get_paragraph_vector(job_pre)
    resume_vector = get_paragraph_vector(resume_pre)

    res = cosine_sim(resume_vector,job_vector)

    st.write(res * 100)


# else:
#     st.write('Button was not clicked.')