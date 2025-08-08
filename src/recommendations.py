import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import load_processed, load_embeddings

def recommend_jobs_for_resume(resume_idx, top_n=5):
    resumes_df, jobs_df = load_processed()
    resume_embs, job_embs = load_embeddings()

    if resume_idx >= resume_embs.shape[0]:
        raise IndexError("resume index out of range")

    query = resume_embs[resume_idx:resume_idx+1]
    sims = cosine_similarity(query, job_embs)[0]

    jobs_df = jobs_df.copy()
    jobs_df['match_percent'] = (sims * 100).round(2)

    # Calculate skill overlap if available
    if 'skills' in jobs_df.columns and 'skills' in resumes_df.columns:
        resume_skills = resumes_df.loc[resume_idx, 'skills']
        if isinstance(resume_skills, str):
            resume_skills = [s.strip() for s in resume_skills.strip("[]").replace("'", "").split(",") if s.strip()]
        jobs_df['skill_overlap'] = jobs_df['skills'].apply(
            lambda s: len(set(
                [x.strip() for x in (s if isinstance(s, list) else str(s).split(","))]
            ) & set(resume_skills))
        )
    else:
        jobs_df['skill_overlap'] = 0

    # Remove duplicates based on job description
    jobs_df = jobs_df.drop_duplicates(subset=['job_description_clean'])

    # Sort by match % and then skill overlap
    jobs_df = jobs_df.sort_values(by=['match_percent', 'skill_overlap'], ascending=False)

    # Only keep relevant columns for output
    cols_to_show = ['Job Title', 'match_percent', 'skill_overlap']
    return jobs_df[cols_to_show].head(top_n)

if __name__ == "__main__":
    top_jobs = recommend_jobs_for_resume(0, top_n=5)
    print(top_jobs.to_string(index=False))
