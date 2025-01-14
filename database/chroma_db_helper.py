import json
import os

import chromadb
import requests
import pandas as pd
import numpy as np
from numpy.linalg import norm





text_chunks = [
    "The sky is blue.",
    "The grass is green.",
    "The sun is shining.",
    "I love chocolate.",
    "Pizza is delicious.",
    "Coding is fun.",
    "Roses are red.",
    "Violets are blue.",
    "Water is essential for life.",
    "The moon orbits the Earth.",
]

def _get_embeddings(text_chunk):
    model_id = 'sentence-transformers/all-MiniLM-L6-v2'

    hf_token = os.environ['hf_token']

    # API endpoint for embedding model
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
    headers = {"Authorization": f"Bearer {hf_token}"}

    response = requests.post(api_url, headers=headers, json={"inputs": text_chunk, "options": {"wait_for_model": True}})

    # load response from embedding model into json format
    embedding = response.json()

    return embedding

def from_text_to_embeddings(text_chunks):
    '''
    Translate sentences into vector embeddings

    Attributes:
        - text_chunks (list): list of example strings

    Returns:
        - embeddings_df (DataFrame): data frame with the columns "text_chunk" and "embeddings"
    '''
    # create new data frame using text chunks list
    embeddings_df = pd.DataFrame(text_chunks).rename(columns={0:"text_chunk"})

    # use the _get_embeddings function to retrieve the embeddings for each of the sentences
    embeddings_df["embeddings"] = embeddings_df["text_chunk"].apply(_get_embeddings)

    # split the embeddings column into individuell columns for each vector dimension
    embeddings_df = embeddings_df['embeddings'].apply(pd.Series)
    embeddings_df["text_chunk"] = text_chunks

    return embeddings_df

def calcaualte_cosine_similiarity(text_chunk, embeddings_df:pd.DataFrame):
    sentence_embedding = _get_embeddings(text_chunk)

    embeddings_df['embeddings_array'] = embeddings_df.apply(lambda row: row.values[:-1], axis=1)
    cos_sim = []

    for index, row in embeddings_df.iterrows():
        A = row.embeddings_arry
        B = sentence_embedding

        cosine = np.dot(A,B) / (norm(A), norm(B))
        cos_sim.append(cosine)

    embeddings_cosine_df = embeddings_df
    embeddings_cosine_df["cos_sim"] = cos_sim
    embeddings_cosine_df.sort_values(by=["cos_sim"], ascending=False)

VECTOR_STORE_PATH = r'../'
COLLECTION_NAME = "chroma_db_collection"
def get_or_create_client_and_collection(VECTOR_STORE_PATH, COLLECTION_NAME):
    # get/create a chroma client
    chroma_client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

    #get or create collection
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


# get embeddings for each of the text chunks
embeddings_df = from_text_to_embeddings(text_chunks)

# save data frame with text chunks and embeddings to csv
embeddings_df.to_csv('../embeddings_df.csv', index=False)

collection = get_or_create_client_and_collection(VECTOR_STORE_PATH,COLLECTION_NAME )

def add_to_collection(embeddings_df:pd.DataFrame):
    embeddings_df['embeddings_array'] = embeddings_df.apply(lambda  row: row.values[:-1], axis=1)
    embeddings_df['embeddings_array'] = embeddings_df['embeddings_array'].apply(lambda x: x.tolist())

    collection.add(
        embeddings=embeddings_df.embeddings_array.to_list(),
        documents=embeddings_df.text_chunk.to_list(),
        ids = list(map(str, embeddings_df.index.tolist()))
    )


def get_all_entries(collection):
    existing_docs = pd.DataFrame(collection.get()).rename(columns={0: "ids", 1:"embeddings", 2:"documents", 3:"metadatas"})
    return existing_docs

def query_vector_database(VECTOR_STORE_PATH, COLLECTION_NAME, query, n=2):
    # query collection
    results = collection.query(
        query_texts=query,
        n_results=n
    )

    print(f"Similarity Search: {n} most similar entries:")
    print(results["documents"])
    return results

# similarity search

add_to_collection(embeddings_df)

exist_docs = get_all_entries(collection)
similar_vector_entries = query_vector_database(VECTOR_STORE_PATH, COLLECTION_NAME, query=["Lilies are white."])
print(f'similar_vector_entries : {similar_vector_entries}')






