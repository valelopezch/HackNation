# 🧠 TalentAI – AI-Powered Job Matching Platform

TalentAI is an AI-driven platform for matching **AI & ML job listings** with qualified candidates.  
It uses **vector similarity search** (Sentence-BERT embeddings + FAISS) combined with TF-IDF and skill-overlap scoring to recommend the best matches for each job or candidate.  
Recruiters can post jobs and instantly see top matches. Candidates can browse best-fit jobs and complete skill validation quizzes.

---

## ✨ Features

- **User roles**:  
  - **Recruiters**: Post new jobs, view matches for each job, and see applicants ranked by validation + match score.  
  - **Candidates**: Sign up, upload CVs (auto-parsed), view recommended jobs, and apply with MCQ skill validation.
- **Matching engine**: Hybrid ranking using:
  - TF-IDF on job/candidate text.
  - Dense embeddings from `paraphrase-multilingual-MiniLM-L12-v2` via [SentenceTransformers](https://www.sbert.net/).
  - Jaccard skill-overlap.
- **Vector store**: Built with FAISS for fast nearest-neighbor search.
- **CSV-backed storage**: Users, jobs, candidates, and applications are persisted in `/data`.
- **Automatic re-indexing**: Adding a new job or candidate automatically rebuilds the relevant FAISS + TF-IDF indexes.
- **Validation quizzes**: Topic-matched MCQs from a CSV dataset.

---

## 📂 Project Structure

```
.
├── app.py                # Main Streamlit application
├── embed.py              # DualMatcher class (TF-IDF + embeddings + FAISS)
├── data_utils.py         # CSV I/O, user/job/candidate CRUD
├── validate.py           # Quiz building and grading
├── data/
│   ├── users.csv
│   ├── recruiters.csv
│   ├── jobs.csv
│   ├── candidates.csv
│   ├── applications.csv
│   └── interview_questions_mcq.csv
├── vector_store/         # Saved FAISS indexes + TF-IDF models
└── requirements.txt
```

---

## 🚀 Quick Start

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/talentai.git
cd talentai
```

### 2️⃣ Install dependencies
It’s recommended to use a Python virtual environment (3.9+).
```bash
pip install -r requirements.txt
```

**Key dependencies**:
- `streamlit`
- `pandas`
- `sentence-transformers`
- `faiss-cpu`
- `scikit-learn`
- `PyPDF2`
- `python-slugify`
- `faker`
- `bcrypt` *(optional, for password hashing)*

### 3️⃣ Run the app
```bash
streamlit run app.py
```

The app will start on `http://localhost:8501`.

---

## 🛠 How It Works

### Data storage
- All data is stored in `/data` as CSVs.  
- Adding jobs/candidates updates these CSVs and triggers re-embedding.

### Matching engine
The [`DualMatcher`](embed.py) combines:
1. **TF-IDF cosine similarity**.
2. **Dense vector similarity** (FAISS inner product on normalized embeddings).
3. **Skill overlap** score.

Final score:
```
score = (w_tfidf * tfidf_sim) +
        (w_emb * embedding_sim) +
        (w_skill * skill_overlap)
```
…minus rule-based penalties for mismatched seniority, YOE, location, etc.

### Embedding refresh
Any time a CSV signature changes, `DualMatcher` rebuilds the affected store (jobs or candidates) automatically.

---

## 👤 User Roles

### Recruiter
- Post a new job via “➕ Post a new job” form.
- View:
  - Top matches from **all candidates**.
  - Top matches from **applicants only**.
- See match scores + validation scores.

### Candidate
- Sign up with a CV (PDF) — key profile fields auto-parsed.
- See **top matching jobs** based on their skills and profile.
- Apply to jobs by completing MCQ validation quizzes.

---

## 📌 Adding Data

### Adding a new job (Recruiter view)
1. Go to recruiter dashboard.
2. Open “➕ Post a new job”.
3. Fill in required fields (`Job title`, `Skills/Tech stack`).
4. Submit → the job is saved to `jobs.csv` and indexed immediately.

### Adding a new candidate
- Done automatically on candidate signup.  
- The profile is saved to `candidates.csv` and embedded instantly.

---

## 🧪 Skill Validation Quizzes

- Stored in `data/interview_questions_mcq.csv`.
- Must have columns:
  ```
  topic,question,options,correct_answer
  ```
- When a candidate applies, quiz questions are matched by job topic.

---

## 🗄 Data Files

- `users.csv` — account credentials + role info.
- `recruiters.csv` — recruiter profiles.
- `jobs.csv` — job postings.
- `candidates.csv` — candidate profiles.
- `applications.csv` — applications + scores.

---

## 🏗 Extending the Project

Ideas for next steps:
- Incremental vector store upserts (avoid full rebuilds).
- Multi-language MCQ datasets.
- Richer CV parsing (NER on experience, education).
- Web deployment on Streamlit Cloud / AWS.

---

## 📜 License
MIT License — free to use and modify for hackathon and educational purposes.


