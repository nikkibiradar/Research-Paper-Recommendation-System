import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_sci_lg
import string
from sklearn.metrics.pairwise import cosine_similarity

st.title("Research Paper Recommender")
    
MODEL_URL = "https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2"
sentence_encoder_layer = hub.KerasLayer(MODEL_URL, input_shape=[], dtype=tf.string, trainable=False, name="use")

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

@st.cache_resource
def get_model():
    with open("model.pkl", 'rb') as file:
        model = pickle.load(file)
    return model

@st.cache_resource
def load_embeddings():
    with open("embeddings.pkl", 'rb') as file:
        embeddings = pickle.load(file)
    return embeddings

df = load_data()
model = get_model()
embed = load_embeddings()
nn = model["model"]
embeddings = embed["embeddings"]

def calculate_user_embedding(abstract):
    user_abstract = sentence_encoder_layer([abstract])
    return user_abstract.numpy()

def find_similar_papers(user_embeddings, k=5):
    dist, indices = nn.kneighbors(X=user_embeddings, n_neighbors=k)
    similar_paper_embeddings = embeddings[indices[0]]
    similar_paper_titles = df['title'].iloc[indices[0]].tolist()
    return indices[0], similar_paper_embeddings, similar_paper_titles

@st.cache_resource
def spacy_tokenizer(sentence):
    punctuations = string.punctuation #list of punctuation to remove from text
    stopwords = list(STOP_WORDS)
    parser = en_core_sci_lg.load()
    parser.max_length = 7000000
    mytokens = parser(sentence)
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ] # transform to lowercase and then split the scentence
    mytokens = [ word for word in mytokens if word not in stopwords and word not in punctuations ] #remove stopsword an punctuation
    mytokens = " ".join([i for i in mytokens]) 
    return mytokens

def main():
    with st.form("recommendation_form"):
        title = st.text_input('Please Enter the Paper Title')
        abstract = st.text_area('Please Enter the Abstract of the Paper')
        submit_button = st.form_submit_button("Recommend Papers!")
   
    col1, col2 = st.columns(2)
    if submit_button:
        if abstract == "":
            abstract = title
        processed_abstract = spacy_tokenizer(abstract)
        user_embeddings = calculate_user_embedding(processed_abstract)
        similar_paper_indices, similar_paper_embeddings, similar_paper_titles = find_similar_papers(user_embeddings)
        col1.header("Recommendations are: ")
        for i, idx in enumerate(similar_paper_indices):
            recommended_paper = df['title'][idx]
            col1.write(f"Recommendation {i + 1}:\n{recommended_paper}\n")
            
        title_embeddings = similar_paper_embeddings[:, :512]
        abstract_embeddings = similar_paper_embeddings[:, 512:]
        similar_paper_combined_embeddings = np.concatenate([title_embeddings, abstract_embeddings], axis=1)
        similarities = cosine_similarity(user_embeddings, similar_paper_combined_embeddings)
        import seaborn as sns
        import matplotlib.pyplot as plt
        similarity_scores = similarities.flatten()
        sns.heatmap([similarity_scores], annot=True, cmap="YlGnBu", xticklabels=similar_paper_titles, yticklabels=["User"], cbar_kws={'label': 'Cosine Similarity'})
        plt.title('Cosine Similarity Between User Abstract and Recommended Papers')
        col2.header("Cosine Similarity: ")
        col2.pyplot(fig=plt, clear_figure=None, use_container_width=True)
            
if __name__ == "__main__":
    main()       