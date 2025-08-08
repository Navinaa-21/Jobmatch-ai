# src/embeddings.py
import os
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

def load_model():
    return SentenceTransformer(MODEL_NAME)

def embed_texts(texts, model, batch_size=64):
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=batch_size)
    return embeddings

def build_and_save_embeddings():
    model = load_model()
    # load processed data
    resumes_df = pd.read_csv(os.path.join(PROCESSED, "resumes_clean.csv"))
    jobs_df = pd.read_csv(os.path.join(PROCESSED, "jobs_clean.csv"))

    # define text fields
    resumes_texts = resumes_df['resume_clean'].fillna("").astype(str).tolist()
    jobs_texts = jobs_df['job_description_clean'].fillna("").astype(str).tolist()

    print("Embedding resumes...")
    resume_embs = embed_texts(resumes_texts, model)
    print("Embedding jobs...")
    job_embs = embed_texts(jobs_texts, model)

    # save numpy arrays and ids
    np.save(os.path.join(MODELS_DIR, "resume_embeddings.npy"), resume_embs)
    np.save(os.path.join(MODELS_DIR, "job_embeddings.npy"), job_embs)

    # save metadata mappings
    with open(os.path.join(MODELS_DIR, "resume_index.pkl"), "wb") as f:
        pickle.dump(resumes_df.index.tolist(), f)
    with open(os.path.join(MODELS_DIR, "job_index.pkl"), "wb") as f:
        pickle.dump(jobs_df.index.tolist(), f)

    print("Saved embeddings to", MODELS_DIR)

if __name__ == "__main__":
    build_and_save_embeddings()
