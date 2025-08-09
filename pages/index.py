# talentai/index.py PISA
from __future__ import annotations
import numpy as np, pandas as pd, duckdb as ddb
from typing import Dict, List, Optional, Tuple
import streamlit as st

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False
    from sklearn.neighbors import NearestNeighbors

from .embed import embed_texts

def _ensure_float32(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

class CandidateIndex:
    """
    Hybrid: DuckDB for meta-filters → shortlist; FAISS (or sklearn) for semantic retrieval.
    """
    def __init__(self, df: pd.DataFrame, text_col="profile_text", id_col="candidate_id"):
        self.df = df.reset_index(drop=True)
        self.text_col, self.id_col = text_col, id_col
        self.vecs = _ensure_float32(embed_texts(self.df[text_col].tolist()))
        self.ids = self.df[id_col].tolist()
        self.id_to_row = {cid: i for i, cid in enumerate(self.ids)}

        if _HAS_FAISS:
            d = self.vecs.shape[1]
            self.index = faiss.IndexFlatIP(d)  # dot on normalized = cosine
            self.index.add(self.vecs)
        else:
            self.index = NearestNeighbors(metric="cosine").fit(self.vecs)

        # DuckDB in-memory catalog for filters
        self.conn = ddb.connect()
        self.conn.register("candidates", self.df)

    def filter_ids(self, region: Optional[str], min_grad_year: Optional[int], availability: Optional[str]) -> List[int]:
        where = []
        if region: where.append(f"region = '{region}'")
        if min_grad_year: where.append(f"grad_year >= {int(min_grad_year)}")
        if availability: where.append(f"availability = '{availability}'")
        where_sql = " AND ".join(where) if where else "TRUE"
        q = f"SELECT {self.id_col} FROM candidates WHERE {where_sql};"
        return [int(x[0]) for x in self.conn.execute(q).fetchall()]

    def search(self, query_text: str, k: int = 20, shortlist_ids: Optional[List[int]] = None) -> List[Tuple[int,float]]:
        q = _ensure_float32(embed_texts([query_text]))  # (1, d)
        if shortlist_ids:
            idxs = np.array([self.id_to_row[i] for i in shortlist_ids], dtype=np.int64)
            sub_vecs = self.vecs[idxs]
            if _HAS_FAISS:
                # brute force on the subset for simplicity
                sims = (sub_vecs @ q.T).ravel()
                top = sims.argsort()[::-1][:k]
                return [(int(self.ids[idxs[i]]), float(sims[i])) for i in top]
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(q, sub_vecs).ravel()
                top = sims.argsort()[::-1][:k]
                return [(int(self.ids[idxs[i]]), float(sims[i])) for i in top]
        else:
            if _HAS_FAISS:
                sims, I = self.index.search(q, k)
                return [(int(self.ids[i]), float(sims[0, j])) for j, i in enumerate(I[0])]
            else:
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(q, self.vecs).ravel()
                top = sims.argsort()[::-1][:k]
                return [(int(self.ids[i]), float(sims[i])) for i in top]

def skill_overlap_score(row_skills: set[str], query_skills: set[str]) -> float:
    if not row_skills or not query_skills: return 0.0
    return len(row_skills & query_skills) / len(row_skills | query_skills)

def hybrid_rank(results: List[Tuple[int,float]], df: pd.DataFrame, query_skills: set[str],
                w_sem=0.7, w_skill=0.3) -> pd.DataFrame:
    rows = []
    for cid, sem in results:
        row = df[df["candidate_id"] == cid].iloc[0]
        s_overlap = skill_overlap_score(set(row["skills"]), query_skills)
        score = w_sem*sem + w_skill*s_overlap
        rows.append({"candidate_id": cid, "score": score, "semantic": sem, "skill_overlap": s_overlap})
    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    return out
