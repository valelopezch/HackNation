import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------- Helpers
def _coalesce_text(*xs):
    return " | ".join([str(x) for x in xs if isinstance(x, str) and x.strip()])

def build_job_corpus(jobs: pd.DataFrame) -> pd.Series:
    # Combine meaningful fields for text matching
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
    """Return penalty in [0, 0.4] to downweight mismatches."""
    penalty = 0.0
    # Years of experience
    try:
        req = float(job.get("yoe", 0))
        got = float(cand_row.get("yoe", 0))
        if got + 0.5 < req:
            penalty += 0.15
    except:
        pass

    # Employment type match (if provided)
    jt = str(job.get("employment_type","")).lower()
    ct = str(cand_row.get("preferred_employment_type","")).lower()
    if jt and ct and jt not in ct:
        penalty += 0.1

    # Seniority alignment (soft)
    js = str(job.get("seniority","")).lower()
    cs = str(cand_row.get("seniority","")).lower()
    if js and cs and js not in cs:
        penalty += 0.08

    # Region/site hint (super soft)
    site = str(job.get("site (remote country)","")).lower()
    loc = str(cand_row.get("location","")).lower()
    if site and loc and site not in loc and "remote" not in site:
        penalty += 0.07

    return min(penalty, 0.4)

class TFIDFMatcher:
    def __init__(self, jobs_df: pd.DataFrame):
        self.jobs = jobs_df.copy()
        self.jobs_corpus = build_job_corpus(self.jobs)
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1,2))
        self.job_X = self.vectorizer.fit_transform(self.jobs_corpus)

    def score_candidate_vs_jobs(self, cand_text: str, cand_row: pd.Series) -> pd.DataFrame:
        q = self.vectorizer.transform([cand_text])
        base = cosine_similarity(q, self.job_X).ravel()  # [n_jobs]
        scores = []
        for i, job in self.jobs.iterrows():
            penalty = _rule_penalties(job, cand_row)
            s = max(0.0, base[i] * (1.0 - penalty))
            scores.append(s)
        out = self.jobs[["job_id","job_title","topic","posted_by"]].copy()
        out["score_match"] = scores
        return out.sort_values("score_match", ascending=False)

    def score_job_vs_candidates(self, job_row: pd.Series, candidates_df: pd.DataFrame, skills_map: dict) -> pd.DataFrame:
        cand_texts = []
        for _, c in candidates_df.iterrows():
            skills_text = skills_map.get(c["candidate_email"], "")
            cand_texts.append(build_candidate_text(c, skills_text))
        Q = self.vectorizer.transform(cand_texts)
        j_idx = self.jobs.index[self.jobs["job_id"] == job_row["job_id"]][0]
        base = cosine_similarity(Q, self.job_X[j_idx])
        base = base.ravel()

        out = candidates_df[["candidate_email","full_name"]].copy()
        scores = []
        for (idx, c), b in zip(candidates_df.iterrows(), base):
            penalty = _rule_penalties(job_row, c)
            scores.append(max(0.0, b * (1.0 - penalty)))
        out["score_match"] = scores
        return out.sort_values("score_match", ascending=False)
