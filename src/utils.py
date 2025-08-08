import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROCESSED = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")

def load_processed():
    """Load the processed resumes and jobs datasets."""
    resumes_df = pd.read_csv(os.path.join(PROCESSED, "resumes_clean.csv"))
    jobs_df = pd.read_csv(os.path.join(PROCESSED, "jobs_clean.csv"))
    return resumes_df, jobs_df

def load_embeddings():
    """Load precomputed embeddings for resumes and jobs."""
    resume_embs = np.load(os.path.join(MODELS_DIR, "resume_embeddings.npy"))
    job_embs = np.load(os.path.join(MODELS_DIR, "job_embeddings.npy"))
    return resume_embs, job_embs
