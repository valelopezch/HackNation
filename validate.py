# validate.py
import hashlib
import random
import time
import ast
from typing import List, Dict, Any, Set, Optional

import pandas as pd

# ------------------------ Helpers & dataset loading ------------------------

def _parse_skills_field(text) -> List[str]:
    """(Kept for compatibility; unused now.)"""
    if not isinstance(text, str):
        return []
    raw = [t.strip() for t in text.replace("/", ",").replace("|", ",").split(",")]
    return [r for r in raw if r]

def extract_global_skill_pool(jobs_df: pd.DataFrame) -> List[str]:
    """(Kept for compatibility; unused by quiz now.)"""
    seen: Set[str] = set()
    pool: List[str] = []
    for s in jobs_df.get("Skills/Tech-stack required", []):
        for x in _parse_skills_field(s):
            if x not in seen:
                seen.add(x)
                pool.append(x)
    return pool

def _seed_from_job(job_row: pd.Series, salt: str = "") -> int:
    """Stable, job-specific seed to make randomness deterministic across reruns."""
    key = f"{job_row.get('job_id','')}|{job_row.get('job_title','')}|{job_row.get('topic','')}|{salt}"
    return int(hashlib.blake2s(key.encode("utf-8"), digest_size=4).hexdigest(), 16)

def _safe_parse_options(obj: Any) -> List[str]:
    """
    'options' in CSV is a literal Python list string; parse robustly.
    If parsing fails, fall back to simple splitting.
    """
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if not isinstance(obj, str):
        return []
    txt = obj.strip()
    try:
        val = ast.literal_eval(txt)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val]
    except Exception:
        pass
    parts = [p.strip().strip("[]'\" ") for p in txt.split(",")]
    return [p for p in parts if p]

def load_mcq_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    req_cols = {"topic", "question", "options", "correct_answer"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"MCQ file is missing columns: {missing}")
    df["topic"] = df["topic"].astype(str)
    df["question"] = df["question"].astype(str)
    df["correct_answer"] = df["correct_answer"].astype(str)
    df["options_parsed"] = df["options"].apply(_safe_parse_options)
    df = df[df["options_parsed"].apply(lambda x: isinstance(x, list) and len(x) >= 2)].copy()
    df = df[df.apply(lambda r: str(r["correct_answer"]) in [str(o) for o in r["options_parsed"]], axis=1)].copy()
    return df.reset_index(drop=True)

# ------------------------ Core quiz API (names unchanged) ------------------------

def build_quiz(
    job_row: pd.Series,
    global_skill_pool: List[str],   # kept for signature compatibility; not used
    *,
    mcq_df: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None,
    include_debug: bool = False,    # ignored; only dataset MCQs are used
) -> List[Dict[str, Any]]:
    """
    Build a deterministic quiz with exactly 5 single-choice questions,
    taken ONLY from the CSV dataset, filtered by the job's exact topic.
    """
    if mcq_df is None or mcq_df.empty:
        return []

    base_seed = _seed_from_job(job_row) if seed is None else seed
    rng = random.Random(base_seed)

    # Filter by exact topic (case-insensitive). If empty, fall back to all.
    job_topic = str(job_row.get("topic", "")).strip()
    pool = mcq_df[mcq_df["topic"].str.lower() == job_topic.lower()]
    if pool.empty:
        pool = mcq_df

    # Pick up to 5 questions deterministically
    n = min(5, len(pool))
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    chosen = [pool.iloc[i] for i in idxs[:n]]

    quiz: List[Dict[str, Any]] = []
    for row in chosen:
        options = list(row["options_parsed"])
        rng.shuffle(options)
        answer_idx = options.index(str(row["correct_answer"]))
        quiz.append(
            {
                "type": "single",
                "question": row["question"],
                "options": options,
                "answer_idx": answer_idx,
                "source_topic": row["topic"],
            }
        )

    return quiz

def default_answers_skeleton(quiz: List[Dict[str, Any]]) -> List[Any]:
    """Produce an answers list with proper shapes matching the quiz."""
    ans: List[Any] = []
    for q in quiz:
        if q["type"] == "single":
            ans.append(None)   # store selected option index (int)
        elif q["type"] == "multi":
            ans.append([])     # (not used now)
        else:
            ans.append(None)
    return ans

def grade_quiz(answers: List[Any], quiz: List[Dict[str, Any]]) -> Dict[str, Any]:
    score = 0
    max_score = len(quiz)
    for ans, q in zip(answers, quiz):
        if q["type"] == "single":
            if ans is not None and int(ans) == int(q["answer_idx"]):
                score += 1
        elif q["type"] == "multi":
            sel = set(ans or [])
            if sel == q.get("answer_set", set()):
                score += 1
    pct = round(100.0 * score / max_score, 1) if max_score else 0.0
    return {
        "score_raw": score,
        "score_max": max_score,
        "score_pct": pct,
        "timestamp": int(time.time()),
    }
