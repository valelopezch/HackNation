import os
import uuid
import pandas as pd
from slugify import slugify
from datetime import datetime

DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users.csv")
RECRUITERS_FILE = os.path.join(DATA_DIR, "recruiters.csv")
JOBS_FILE = os.path.join(DATA_DIR, "jobs.csv")
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates.csv")
APPLICATIONS_FILE = os.path.join(DATA_DIR, "applications.csv")

# -----------------------------
# Loading & integrity utilities
# -----------------------------

def _iso_now() -> str:
    return datetime.utcnow().isoformat()

def _ensure_file(path: str, columns: list[str]):
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def load_users() -> pd.DataFrame:
    _ensure_file(USERS_FILE, ["email","password_hash","role","full_name","location","created_at"])
    df = pd.read_csv(USERS_FILE).fillna("")
    # normalize columns
    if "role" in df.columns:
        df["role"] = df["role"].str.strip().str.lower()
    return df

def load_recruiters() -> pd.DataFrame:
    _ensure_file(RECRUITERS_FILE, ["email","company_name","company_site","bio","posted_jobs"])
    return pd.read_csv(RECRUITERS_FILE).fillna("")

def ensure_job_ids(df_jobs: pd.DataFrame) -> pd.DataFrame:
    df = df_jobs.copy()
    if "job_id" not in df.columns:
        df["job_id"] = df.apply(lambda r: slugify(f"{r.get('job_title','job')}-{r.name}"), axis=1)
    # assure string type
    df["job_id"] = df["job_id"].astype(str)
    return df

def ensure_posted_by(df_jobs: pd.DataFrame, default_owner: str = "recruiter@demo") -> pd.DataFrame:
    df = df_jobs.copy()
    if "posted_by" not in df.columns:
        df["posted_by"] = default_owner
    df["posted_by"] = df["posted_by"].fillna(default_owner).astype(str)
    return df

def load_jobs() -> pd.DataFrame:
    _ensure_file(JOBS_FILE, [
        "job_id","posted_by","topic","job_title","site (remote country)","tasks",
        "Perks/Benefits","Skills/Tech-stack required","Educational requirements",
        "seniority","yoe","employment_type","extra_info","created_at"
    ])
    jobs = pd.read_csv(JOBS_FILE).fillna("")
    jobs = ensure_job_ids(jobs)
    jobs = ensure_posted_by(jobs)
    return jobs

def load_candidates() -> pd.DataFrame:
    _ensure_file(CANDIDATES_FILE, [
        "candidate_email","full_name","candidate_title","about","location",
        "preferred_employment_type","yoe","seniority","skills","created_at"
    ])
    cands = pd.read_csv(CANDIDATES_FILE).fillna("")
    if "candidate_email" not in cands.columns:
        # create deterministic demo emails if missing (safety net)
        cands["candidate_email"] = [f"cand{i}@demo" for i in range(len(cands))]
    return cands

def load_applications() -> pd.DataFrame:
    _ensure_file(APPLICATIONS_FILE, [
        "app_id","job_id","candidate_email","status","score_validation","score_match","created_at"
    ])
    apps = pd.read_csv(APPLICATIONS_FILE).fillna("")
    return apps

# -----------------------------
# Authentication helpers
# -----------------------------

def _bcrypt_check(password_plain: str, hashed: str) -> bool:
    """
    Try bcrypt if available. If not, fall back to plaintext comparison (hackathon mode).
    """
    if not hashed:
        return False
    try:
        import bcrypt
        # If the stored "hash" is actually plaintext (for quick tests), compare directly.
        if not hashed.startswith("$2a$") and not hashed.startswith("$2b$") and not hashed.startswith("$2y$"):
            return password_plain == hashed
        return bcrypt.checkpw(password_plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        # Fallback: plaintext comparison
        return password_plain == hashed

def authenticate_user(email: str, password: str) -> dict | None:
    users = load_users()
    row = users[users["email"].str.lower() == str(email).strip().lower()]
    if row.empty:
        return None
    r = row.iloc[0].to_dict()
    if _bcrypt_check(password, str(r.get("password_hash",""))):
        return {
            "email": r.get("email",""),
            "role": r.get("role",""),
            "full_name": r.get("full_name",""),
            "location": r.get("location","")
        }
    return None

# -----------------------------
# Applications I/O
# -----------------------------

def save_application(job_id: str, candidate_email: str, score_validation: float, score_match: float, status: str = "applied") -> dict:
    apps = load_applications()
    app = {
        "app_id": str(uuid.uuid4())[:8],
        "job_id": str(job_id),
        "candidate_email": str(candidate_email),
        "status": str(status),
        "score_validation": float(score_validation),
        "score_match": float(score_match),
        "created_at": _iso_now()
    }
    apps = pd.concat([apps, pd.DataFrame([app])], ignore_index=True)
    apps.to_csv(APPLICATIONS_FILE, index=False)
    return app

def apps_for_job(job_id: str) -> pd.DataFrame:
    apps = load_applications()
    return apps[apps["job_id"].astype(str) == str(job_id)].copy()

def apps_for_candidate(email: str) -> pd.DataFrame:
    apps = load_applications()
    return apps[apps["candidate_email"].str.lower() == str(email).lower()].copy()

# -----------------------------
# Convenience helpers
# -----------------------------

def candidate_skills_map(cands: pd.DataFrame) -> dict[str, str]:
    """
    Build a dict email -> skills_text from candidates.csv.
    """
    out = {}
    for _, r in cands.iterrows():
        out[str(r["candidate_email"])] = str(r.get("skills",""))
    return out

def global_skill_pool_from_candidates(cands: pd.DataFrame) -> list[str]:
    """
    Create a global pool of skills from the candidates' 'skills' column (comma-separated).
    """
    all_text = ", ".join([str(x) for x in cands.get("skills","").tolist()]).split(",")
    return sorted(set(s.strip() for s in all_text if str(s).strip()))