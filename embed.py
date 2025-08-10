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

import hashlib, json, time
from pathlib import Path

# sign and manifest helpers

def df_sha256(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def config_sha256(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

def store_dir_for(root: str, kind: str, data_sig: str, cfg_sig: str) -> Path:
    p = Path(root) / f"{kind}_{data_sig[:8]}__cfg_{cfg_sig[:8]}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def write_manifest(dirpath: Path, meta: dict):
    meta = {**meta, "updated_at": int(time.time())}
    (dirpath / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def read_manifest(dirpath: Path) -> dict | None:
    p = dirpath / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None

def build_candidate_corpus(cands: pd.DataFrame, skills_map: Dict[str, str]) -> pd.Series:
    def _row_text(r):
        skills_text = skills_map.get(r.get("candidate_email",""), "")
        return build_candidate_text(r, skills_text)
    return cands.apply(_row_text, axis=1)

def needs_rebuild(dirpath: Path, expect: dict) -> bool:
    man = read_manifest(dirpath)
    if not man:
        return True
    keys = ["sha256","rows","model_name","tfidf_min_df","tfidf_ngram"]
    return any(man.get(k) != expect.get(k) for k in keys)

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

class DualMatcher:
    """
    Índices persistentes y separados:
      - Jobs store:   vectorizador y FAISS entrenados sobre 'jobs'
      - Cands store:  vectorizador y FAISS entrenados sobre 'candidates'
    """
    def __init__(
        self,
        jobs_df: pd.DataFrame,
        cands_df: pd.DataFrame,
        skills_map: Dict[str, str] | None = None,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        store_root: str = "./vector_store",
    ):
        # Datos base (índices posicionales limpios)
        self.debug = {"jobs": {}, "cands": {}}

        self.jobs  = jobs_df.copy().reset_index(drop=True)
        self.cands = cands_df.copy().reset_index(drop=True)
        self.skills_map = skills_map or {}

        # Config y firmas
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.cfg = {"model_name": model_name, "tfidf_min_df": 1, "tfidf_ngram": (1,2)}
        self.cfg_sig   = config_sha256(self.cfg)
        self.jobs_sig  = df_sha256(self.jobs)
        self.cands_sig = df_sha256(self.cands)

        self.root = Path(store_root); self.root.mkdir(parents=True, exist_ok=True)

        # ==== JOBS STORE ====
        self.jobs_dir = store_dir_for(store_root, "jobs", self.jobs_sig, self.cfg_sig)
        self.jvec_path   = self.jobs_dir / "tfidf_vectorizer.joblib"
        self.jX_path     = self.jobs_dir / "jobs_tfidf.npy"
        self.jfaiss_path = self.jobs_dir / "jobs_faiss.index"

        jobs_corpus = build_job_corpus(self.jobs)
        # TF-IDF (jobs)
        if self.jvec_path.exists() and self.jX_path.exists():
            self.jvec = joblib.load(self.jvec_path)
            self.job_X = np.load(self.jX_path)
            if self.job_X.shape[0] != len(self.jobs):
                # rebuild si hay desalineación
                self.jvec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
                self.job_X = self.jvec.fit_transform(jobs_corpus).toarray().astype(np.float32)
                joblib.dump(self.jvec, self.jvec_path); np.save(self.jX_path, self.job_X)
        else:
            self.jvec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
            self.job_X = self.jvec.fit_transform(jobs_corpus).toarray().astype(np.float32)
            joblib.dump(self.jvec, self.jvec_path); np.save(self.jX_path, self.job_X)

        self.debug["jobs"]["tfidf_loaded"] = self.jvec_path.exists() and self.jX_path.exists() and self.job_X.shape[0] == len(self.jobs)

        # FAISS (jobs)
        if self.jfaiss_path.exists():
            self.jfaiss = faiss.read_index(str(self.jfaiss_path))
            if self.jfaiss.ntotal != len(self.jobs):
                emb = self.model.encode(jobs_corpus.tolist(), normalize_embeddings=True).astype(np.float32)
                self.jfaiss = faiss.IndexFlatIP(emb.shape[1]); self.jfaiss.add(emb); faiss.write_index(self.jfaiss, str(self.jfaiss_path))
        else:
            emb = self.model.encode(jobs_corpus.tolist(), normalize_embeddings=True).astype(np.float32)
            self.jfaiss = faiss.IndexFlatIP(emb.shape[1]); self.jfaiss.add(emb); faiss.write_index(self.jfaiss, str(self.jfaiss_path))

        write_manifest(self.jobs_dir, {
            "sha256": self.jobs_sig, "rows": int(len(self.jobs)),
            "model_name": self.model_name, "tfidf_min_df": 1, "tfidf_ngram": (1,2),
            "faiss_ntotal": int(self.jfaiss.ntotal)
        })

        self.debug["jobs"]["faiss_loaded"] = self.jfaiss_path.exists() and self.jfaiss.ntotal == len(self.jobs)

        # ==== CANDIDATES STORE ====
        self.cands_dir = store_dir_for(store_root, "cands", self.cands_sig, self.cfg_sig)
        self.cvec_path   = self.cands_dir / "tfidf_vectorizer.joblib"
        self.cX_path     = self.cands_dir / "cands_tfidf.npy"
        self.cfaiss_path = self.cands_dir / "cands_faiss.index"

        cands_corpus = build_candidate_corpus(self.cands, self.skills_map)

        # TF-IDF (cands)
        if self.cvec_path.exists() and self.cX_path.exists():
            self.cvec = joblib.load(self.cvec_path)
            self.cand_X = np.load(self.cX_path)
            if self.cand_X.shape[0] != len(self.cands):
                self.cvec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
                self.cand_X = self.cvec.fit_transform(cands_corpus).toarray().astype(np.float32)
                joblib.dump(self.cvec, self.cvec_path); np.save(self.cX_path, self.cand_X)
        else:
            self.cvec = TfidfVectorizer(min_df=1, ngram_range=(1,2))
            self.cand_X = self.cvec.fit_transform(cands_corpus).toarray().astype(np.float32)
            joblib.dump(self.cvec, self.cvec_path); np.save(self.cX_path, self.cand_X)

        self.debug["cands"]["tfidf_loaded"] = self.cvec_path.exists() and self.cX_path.exists() and self.cand_X.shape[0] == len(self.cands)

        # FAISS (cands)
        if self.cfaiss_path.exists():
            self.cfaiss = faiss.read_index(str(self.cfaiss_path))
            if self.cfaiss.ntotal != len(self.cands):
                cemb = self.model.encode(cands_corpus.tolist(), normalize_embeddings=True).astype(np.float32)
                self.cfaiss = faiss.IndexFlatIP(cemb.shape[1]); self.cfaiss.add(cemb); faiss.write_index(self.cfaiss, str(self.cfaiss_path))
        else:
            cemb = self.model.encode(cands_corpus.tolist(), normalize_embeddings=True).astype(np.float32)
            self.cfaiss = faiss.IndexFlatIP(cemb.shape[1]); self.cfaiss.add(cemb); faiss.write_index(self.cfaiss, str(self.cfaiss_path))

        self.debug["cands"]["faiss_loaded"] = self.cfaiss_path.exists() and self.cfaiss.ntotal == len(self.cands)

        write_manifest(self.cands_dir, {
            "sha256": self.cands_sig, "rows": int(len(self.cands)),
            "model_name": self.model_name, "tfidf_min_df": 1, "tfidf_ngram": (1,2),
            "faiss_ntotal": int(self.cfaiss.ntotal)
        })

    def cache_status(self):
        return {
            "jobs_dir": str(self.jobs_dir),
            "cands_dir": str(self.cands_dir),
            "jobs_manifest": read_manifest(self.jobs_dir),
            "cands_manifest": read_manifest(self.cands_dir),
            "status": self.debug,
        }

    # ---------- Candidate -> Jobs ----------
    def score_candidate_vs_jobs(
        self,
        cand_text: str,
        cand_row: pd.Series,
        cand_skills: Set[str],
        w_tfidf=0.4, w_emb=0.4, w_skill=0.2,
        top_k: int = 20
    ) -> pd.DataFrame:
        # TF-IDF (usar vectorizador de jobs)
        q_tfidf = self.jvec.transform([cand_text]).toarray().astype(np.float32)
        sim_tfidf = cosine_similarity(q_tfidf, self.job_X).ravel()

        # Embeddings via FAISS (jobs)
        q_emb = self.model.encode([cand_text], normalize_embeddings=True).astype(np.float32)
        sims_emb, idxs = self.jfaiss.search(q_emb, min(top_k, len(self.jobs)))
        sim_emb = np.zeros(len(self.jobs), dtype=np.float32)
        for s, idx in zip(sims_emb[0], idxs[0]):
            if 0 <= idx < len(sim_emb): sim_emb[idx] = s

        scores = []
        for pos, job in self.jobs.iterrows():
            penalty = _rule_penalties(job, cand_row)
            job_sk = set(job.get("skills", []))
            score = (w_tfidf*sim_tfidf[pos] + w_emb*sim_emb[pos] + w_skill*skill_overlap_score(cand_skills, job_sk)) * (1 - penalty)
            scores.append(score)

        out = self.jobs[["job_id","job_title","topic"]].copy()
        out["score_match"] = scores
        return out.sort_values("score_match", ascending=False).head(top_k)

    # ---------- Job -> Candidates ----------
    def score_job_vs_candidates(
        self,
        job_row: pd.Series,
        candidates_df: pd.DataFrame | None,
        skills_map: Dict[str,str],
        w_tfidf=0.4, w_emb=0.4, w_skill=0.2,
        top_k: int = 20
    ) -> pd.DataFrame:
        # Texto del job
        jtxt = build_job_corpus(pd.DataFrame([job_row])).iloc[0]

        # TF-IDF (usar vectorizador de cands)
        q_tfidf = self.cvec.transform([jtxt]).toarray().astype(np.float32)
        sim_tfidf = cosine_similarity(q_tfidf, self.cand_X).ravel()

        # Embeddings via FAISS (cands)
        q_emb = self.model.encode([jtxt], normalize_embeddings=True).astype(np.float32)
        sims_emb, idxs = self.cfaiss.search(q_emb, min(top_k*5, len(self.cands)))
        sim_emb = np.zeros(len(self.cands), dtype=np.float32)
        for s, idx in zip(sims_emb[0], idxs[0]):
            if 0 <= idx < len(sim_emb): sim_emb[idx] = s

        # Mezcla + reglas
        scores = []
        for pos, c in self.cands.iterrows():
            c_sk_text = skills_map.get(c.get("candidate_email",""), "")
            c_sk = set(s.strip() for s in str(c_sk_text).split(",") if s.strip()) if isinstance(c.get("skills",""), str) else set(c.get("skills", []))
            score = (w_tfidf*sim_tfidf[pos] + w_emb*sim_emb[pos] + w_skill*skill_overlap_score(c_sk, set(job_row.get("skills", [])))) * (1 - _rule_penalties(job_row, c))
            scores.append(score)

        out = self.cands[["candidate_email","full_name"]].copy()
        out["score_match"] = scores

        # Si se pasa un subset (appl_cands), filtramos
        if candidates_df is not None and len(candidates_df) < len(self.cands):
            allowed = set(str(e).lower() for e in candidates_df["candidate_email"].tolist())
            out = out[out["candidate_email"].str.lower().isin(allowed)]

        return out.sort_values("score_match", ascending=False).head(top_k)
