from __future__ import annotations
import os
import numpy as np
import pandas as pd
from typing import Dict, Set, Optional, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import faiss
from sentence_transformers import SentenceTransformer

# -------- Helpers
def _coalesce_text(*xs):
    return " | ".join([str(x) for x in xs if isinstance(x, str) and x.strip()])

def build_job_corpus(jobs: pd.DataFrame) -> pd.Series:
    return jobs.apply(lambda r: _coalesce_text(
        r.get("job_title",""),
        r.get("topic",""),
        r.get("tasks",""),
        r.get("Skills/Tech-stack required",""),
        r.get("Educational requirements",""),
        r.get("extra_info",""),
    ), axis=1)

def build_candidate_text(cand_row: pd.Series, skills_text: str) -> str:
    return _coalesce_text(
        cand_row.get("candidate_title",""),
        cand_row.get("about",""),
        skills_text
    )

def _rule_penalties(job, cand_row) -> float:
    penalty = 0.0
    # Years of experience
    try:
        req = float(job.get("yoe", 0))
        got = float(cand_row.get("yoe", 0))
        if got + 0.5 < req:
            penalty += 0.15
    except:
        pass
    # Employment type
    jt = str(job.get("employment_type","")).lower()
    ct = str(cand_row.get("preferred_employment_type","")).lower()
    if jt and ct and jt not in ct:
        penalty += 0.1
    # Seniority
    js = str(job.get("seniority","")).lower()
    cs = str(cand_row.get("seniority","")).lower()
    if js and cs and js not in cs:
        penalty += 0.08
    # Region/site
    site = str(job.get("site (remote country)","")).lower()
    loc = str(cand_row.get("location","")).lower()
    if site and loc and site not in loc and "remote" not in site:
        penalty += 0.07
    return min(penalty, 0.4)

def skill_overlap_score(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

class HybridMatcher:
    def __init__(
        self,
        jobs_df: pd.DataFrame,
        skills_map: Dict[str, str] = None,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        store_dir: str = "./vector_store"
    ):
        self.jobs = jobs_df.copy()
        self.jobs_corpus = build_job_corpus(self.jobs)
        self.skills_map = skills_map or {}
        self.model_name = model_name
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

        # Paths para cache
        self.tfidf_path = os.path.join(store_dir, "tfidf_vectorizer.joblib")
        self.tfidf_mat_path = os.path.join(store_dir, "jobs_tfidf.npy")
        self.faiss_index_path = os.path.join(store_dir, "jobs_faiss.index")

        # --- TF-IDF ---
        if os.path.exists(self.tfidf_path) and os.path.exists(self.tfidf_mat_path):
            self.vectorizer = joblib.load(self.tfidf_path)
            self.job_X = np.load(self.tfidf_mat_path)
        else:
            self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
            self.job_X = self.vectorizer.fit_transform(self.jobs_corpus).toarray().astype(np.float32)
            joblib.dump(self.vectorizer, self.tfidf_path)
            np.save(self.tfidf_mat_path, self.job_X)

        # --- Embeddings ---
        self.model = SentenceTransformer(model_name)
        if os.path.exists(self.faiss_index_path):
            self.faiss_index = faiss.read_index(self.faiss_index_path)
        else:
            job_emb = self.model.encode(self.jobs_corpus.tolist(), normalize_embeddings=True)
            dim = job_emb.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(job_emb.astype(np.float32))
            faiss.write_index(self.faiss_index, self.faiss_index_path)

    def score_candidate_vs_jobs(
        self,
        cand_text: str,
        cand_row: pd.Series,
        cand_skills: Set[str],
        w_tfidf=0.4, w_emb=0.4, w_skill=0.2,
        top_k: int = 20
    ) -> pd.DataFrame:
        # TF-IDF
        q_tfidf = self.vectorizer.transform([cand_text]).toarray().astype(np.float32)
        sim_tfidf = cosine_similarity(q_tfidf, self.job_X).ravel()

        # Embeddings vía FAISS
        q_emb = self.model.encode([cand_text], normalize_embeddings=True).astype(np.float32)
        sims_emb, idxs = self.faiss_index.search(q_emb, top_k)
        sim_emb = np.zeros(len(self.jobs))
        for s, idx in zip(sims_emb[0], idxs[0]):
            sim_emb[idx] = s

        # Mezcla con skills y reglas
        scores = []
        for i, job in self.jobs.iterrows():
            penalty = _rule_penalties(job, cand_row)
            skill_ov = skill_overlap_score(cand_skills, set(job.get("skills", [])))
            score = (w_tfidf*sim_tfidf[i] + w_emb*sim_emb[i] + w_skill*skill_ov) * (1 - penalty)
            scores.append(score)

        out = self.jobs[["job_id","job_title","topic"]].copy()
        out["score_match"] = scores
        return out.sort_values("score_match", ascending=False).head(top_k)

    def score_job_vs_candidates(
        self,
        job_row: pd.Series,
        candidates_df: pd.DataFrame,
        skills_map: Dict[str,str],
        w_tfidf=0.4, w_emb=0.4, w_skill=0.2,
        top_k: int = 20
    ) -> pd.DataFrame:
        # Prepara corpus de candidatos
        cand_texts = []
        cand_skills_list = []
        for _, c in candidates_df.iterrows():
            skills_text = skills_map.get(c["candidate_email"], "")
            cand_texts.append(build_candidate_text(c, skills_text))
            cand_skills_list.append(set(c.get("skills", [])))

        # TF-IDF
        Q_tfidf = self.vectorizer.transform(cand_texts).toarray().astype(np.float32)
        j_idx = self.jobs.index[self.jobs["job_id"] == job_row["job_id"]][0]
        sim_tfidf = cosine_similarity(Q_tfidf, self.job_X[j_idx].reshape(1, -1)).ravel()

        # Embeddings vía FAISS
        q_emb = self.model.encode([build_job_corpus(pd.DataFrame([job_row])).iloc[0]],
                                  normalize_embeddings=True).astype(np.float32)
        cand_emb = self.model.encode(cand_texts, normalize_embeddings=True).astype(np.float32)
        # Similaridad coseno manual
        sim_emb = np.sum(cand_emb * q_emb, axis=1)

        # Mezcla con skills y reglas
        scores = []
        for (idx, c), s_tfidf, s_emb, s_skills in zip(candidates_df.iterrows(), sim_tfidf, sim_emb, cand_skills_list):
            penalty = _rule_penalties(job_row, c)
            skill_ov = skill_overlap_score(s_skills, set(job_row.get("skills", [])))
            score = (w_tfidf*s_tfidf + w_emb*s_emb + w_skill*skill_ov) * (1 - penalty)
            scores.append(score)

        out = candidates_df[["candidate_email","full_name"]].copy()
        out["score_match"] = scores
        return out.sort_values("score_match", ascending=False).head(top_k)
