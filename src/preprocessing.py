# src/preprocessing.py
import pandas as pd
import re
import os
import json
from tqdm import tqdm
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger"])

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
os.makedirs(PROCESSED, exist_ok=True)

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # replace non-alphanumeric (keep + # . for versions) with space
    text = re.sub(r"[^a-zA-Z0-9\+\#\.\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_raw_resumes(filename="UpdatedResumeDataSet.csv"):
    path = os.path.join(RAW, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put your resume CSV into data/raw/")
    df = pd.read_csv(path)
    # Normalize common column names
    if "Resume" not in df.columns:
        # some datasets use 'Resume Text' or 'resume'
        for c in df.columns:
            if "resume" in c.lower():
                df = df.rename(columns={c: "Resume"})
                break
    return df

def load_raw_jobs(filename="job_descriptions.csv"):
    path = os.path.join(RAW, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Put your jobs CSV into data/raw/")
    df = pd.read_csv(path)
    # Normalize column names
    cols = {c: c for c in df.columns}
    for c in df.columns:
        if "job" in c.lower() and "title" in c.lower():
            cols[c] = "Job Title"
        if "description" in c.lower():
            cols[c] = "Job Description"
        if "skill" in c.lower() or "key skills" in c.lower():
            cols[c] = "Key Skills"
    df = df.rename(columns=cols)
    return df

def build_skills_vocab(jobs_df):
    skills = set()
    if "Key Skills" in jobs_df.columns:
        for entry in jobs_df["Key Skills"].fillna(""):
            # many datasets list skills comma separated
            parts = re.split(r"[;,|]", str(entry))
            for p in parts:
                p = p.strip().lower()
                if p:
                    skills.add(p)
    # Optionally, add some common tech skills
    extra = ["python","java","c++","c#","sql","nosql","postgresql","mongodb","docker","kubernetes",
             "tensorflow","pytorch","scikit-learn","aws","azure","gcp","react","node.js","django","flask"]
    for e in extra:
        skills.add(e)
    return sorted(skills)

def create_phrase_matcher(nlp, skills_list):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(s) for s in skills_list if s.strip()]
    matcher.add("SKILL", patterns)
    return matcher

def extract_skills_from_text(text, matcher):
    doc = nlp(text)
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        span = doc[start:end].text.strip().lower()
        found.add(span)
    # fallback: simple token check for single-word skills (e.g., python, aws)
    tokens = {t.text.lower() for t in doc if t.is_alpha}
    for t in ["python","java","sql","aws","azure","gcp","docker","kubernetes"]:
        if t in tokens:
            found.add(t)
    return sorted(found)

def preprocess_and_save(resume_fname="UpdatedResumeDataSet.csv", job_fname="job_descriptions.csv"):
    print("Loading raw data...")
    resumes = load_raw_resumes(resume_fname)
    jobs = load_raw_jobs(job_fname)

    print("Cleaning resume texts...")
    # Clean resumes
    if "Resume" not in resumes.columns:
        raise ValueError("No 'Resume' column found in resume dataset.")
    resumes["resume_clean"] = resumes["Resume"].apply(clean_text)

    print("Cleaning job descriptions...")
    jd_col = "Job Description" if "Job Description" in jobs.columns else jobs.columns[0]
    jobs["job_description_clean"] = jobs.get("Job Description", jobs[jd_col]).astype(str).apply(clean_text)

    print("Building skills vocabulary from jobs...")
    skills_vocab = build_skills_vocab(jobs)
    # save vocab for later
    with open(os.path.join(PROCESSED, "skills_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(skills_vocab, f, indent=2)

    print("Creating PhraseMatcher and extracting skills from resumes...")
    matcher = create_phrase_matcher(nlp, skills_vocab)

    resumes["skills"] = resumes["resume_clean"].apply(lambda t: extract_skills_from_text(t, matcher))
    # for jobs, normalize Key Skills column to list
    if "Key Skills" in jobs.columns:
        jobs["skills"] = jobs["Key Skills"].fillna("").apply(
            lambda x: [s.strip().lower() for s in re.split(r"[;,|]", str(x)) if s.strip()]
        )
    else:
        # try to extract from description
        jobs["skills"] = jobs["job_description_clean"].apply(lambda t: extract_skills_from_text(t, matcher))

    # Save cleaned files
    resumes_out = os.path.join(PROCESSED, "resumes_clean.csv")
    jobs_out = os.path.join(PROCESSED, "jobs_clean.csv")
    resumes.to_csv(resumes_out, index=False)
    jobs.to_csv(jobs_out, index=False)
    print("Saved:", resumes_out, jobs_out)
    print("Also saved skills_vocab.json in data/processed/")

if __name__ == "__main__":
    preprocess_and_save()
