import sys
import os
# add project root to sys.path so `src` can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import load_processed, load_embeddings
from sklearn.metrics.pairwise import cosine_similarity

def match_score_by_index(resume_idx, job_idx):
    resume_embs, job_embs = load_embeddings()
    if resume_idx >= resume_embs.shape[0] or job_idx >= job_embs.shape[0]:
        raise IndexError("Index out of range for embeddings.")
    score = float(cosine_similarity([resume_embs[resume_idx]], [job_embs[job_idx]])[0][0])
    return round(score * 100, 2)

if __name__ == "__main__":
    resumes_df, jobs_df = load_processed()
    s = match_score_by_index(0, 0)
    print(f"Match Score (resume 0 vs job 0): {s}%")
