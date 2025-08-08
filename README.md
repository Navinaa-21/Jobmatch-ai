# JOBMATCH-AI

JOBMATCH-AI is a Machine Learning project that matches resumes with job descriptions using NLP and semantic similarity.  
It uses **Sentence-BERT embeddings** to find the best job recommendations for a given resume.

---

## 📌 Features
- Preprocess resumes and job descriptions (cleaning, tokenizing, skill extraction)
- Generate embeddings using `all-MiniLM-L6-v2` SentenceTransformer
- Match resumes to jobs using cosine similarity
- Recommend top jobs with highest similarity scores
- Skips preprocessing/embedding steps if files already exist (fast startup on new systems)

