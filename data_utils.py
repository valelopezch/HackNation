import os
import uuid
import pandas as pd
from slugify import slugify
from datetime import datetime, timezone, timedelta
from faker import Faker
import re
import random

fake = Faker()

DATA_DIR = "data"
USERS_FILE = os.path.join(DATA_DIR, "users_template.csv")
RECRUITERS_FILE = os.path.join(DATA_DIR, "recruiters_template.csv")
JOBS_FILE = os.path.join(DATA_DIR, "jobs_template.csv")
CANDIDATES_FILE = os.path.join(DATA_DIR, "candidates_template.csv")
APPLICATIONS_FILE = os.path.join(DATA_DIR, "applications_template.csv")

# -----------------------------
# Loading & integrity utilities
# -----------------------------

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

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

# -----------------------------
# SIGNUP HELPERS 
# -----------------------------

def _bcrypt_hash(password_plain: str) -> str:
    """
    Return a bcrypt hash if bcrypt is available, else return the plaintext
    (ONLY for hackathon/demo). In production, require bcrypt.
    """
    if not password_plain:
        return ""
    try:
        import bcrypt
        salt = bcrypt.gensalt(rounds=12)
        return bcrypt.hashpw(password_plain.encode("utf-8"), salt).decode("utf-8")
    except Exception:
        # Fallback (demo only)
        return password_plain

def email_exists(email: str) -> bool:
    users = load_users()
    if users.empty:
        return False
    return users["email"].str.lower().eq(str(email).lower()).any()

def create_user(email: str, password: str, role: str, full_name: str = "", location: str = "") -> dict:
    """
    Create a new user in users.csv. Returns the created row as a dict.
    """
    if role not in {"candidate", "recruiter"}:
        raise ValueError("role must be 'candidate' or 'recruiter'")
    if email_exists(email):
        raise ValueError("email already exists")

    users = load_users()
    new_row = {
        "email": email.strip(),
        "password_hash": _bcrypt_hash(password),
        "role": role,
        "full_name": full_name or "",
        "location": location or "",
        "created_at": _iso_now()
    }
    users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
    users.to_csv(USERS_FILE, index=False)
    return new_row

def upsert_candidate_profile(
    candidate_email: str,
    full_name: str = "",
    candidate_title: str = "",
    about: str = "",
    location: str = "",
    preferred_employment_type: str = "",
    yoe: int | float | str = "",
    seniority: str = "",
    skills: str = ""
) -> dict:
    cands = load_candidates()
    idx = cands.index[cands["candidate_email"].str.lower() == str(candidate_email).lower()]
    row = {
        "candidate_email": candidate_email,
        "full_name": full_name,
        "candidate_title": candidate_title,
        "about": about,
        "location": location,
        "preferred_employment_type": preferred_employment_type,
        "yoe": yoe,
        "seniority": seniority,
        "skills": skills,
        "created_at": _iso_now()
    }
    if len(idx) > 0:
        # update
        for k, v in row.items():
            cands.loc[idx, k] = v
    else:
        # insert
        cands = pd.concat([cands, pd.DataFrame([row])], ignore_index=True)
    cands.to_csv(CANDIDATES_FILE, index=False)
    return row

def upsert_recruiter_profile(
    email: str,
    company_name: str = "",
    company_site: str = "",
    bio: str = "",
    posted_jobs: int | str = ""
) -> dict:
    recs = load_recruiters()
    idx = recs.index[recs["email"].str.lower() == str(email).lower()]
    row = {
        "email": email,
        "company_name": company_name,
        "company_site": company_site,
        "bio": bio,
        "posted_jobs": posted_jobs if posted_jobs != "" else 0
    }
    if len(idx) > 0:
        for k, v in row.items():
            recs.loc[idx, k] = v
    else:
        recs = pd.concat([recs, pd.DataFrame([row])], ignore_index=True)
    recs.to_csv(RECRUITERS_FILE, index=False)
    return row

## Handle csv files (pdf): 1. save and read pdf 2. obtain data
def calculate_yoe(text):
    matches = re.findall(r'(\d+)\s*(?:\+?\s*)?(?:years?|yrs?)', text, flags=re.IGNORECASE)
    years = [int(m) for m in matches]
    return max(years) if years else 0

def infer_seniority(yoe):
    if yoe < 2:
        return "Junior"
    elif yoe < 5:
        return "Mid-level"
    elif yoe < 10:
        return "Senior"
    else:
        return "Lead/Principal"

def extract_candidate_title(text):
    match = re.search(r'(?:Title|Position|Role|Designation)\s*[:-]?\s*(.+)', text, re.IGNORECASE)
    if match:
        return match.group(1).split('\n')[0].strip()

    # Try from first few lines
    lines = text.strip().split("\n")
    for line in lines[:5]:
        if any(word.lower() in line.lower() for word in 
               ["engineer", "scientist", "developer", "manager", "analyst"]):
            return line.strip()
    return ""

def extract_about(text):
    about = re.sub(r'\s+', ' ', text.strip())
    return about[:500] + ("..." if len(about) > 500 else "")

def extract_employment_type(text):
    types = ["full-time", "part-time", "contract", "internship", "freelance", "temporary"]
    for t in types:
        if re.search(t, text, re.IGNORECASE):
            return t.capitalize()
    return ""

def extract_location(text):
    # Try to find patterns like "City, Country"
    match = re.search(r'([A-Z][a-zA-Z\s]+,\s*[A-Z][a-zA-Z]+)', text)
    if match:
        return match.group(1).strip()
    return "NA"

def random_created_at():
    start = datetime.now() - timedelta(days=365*2)
    end = datetime.now()
    return (start + (end - start) * random.random()).isoformat()

def extract_cv_fields(cv_text, name, email):
    location = extract_location(cv_text)
    created_at = random_created_at()

    yoe = calculate_yoe(cv_text)
    seniority = infer_seniority(yoe)
    title = extract_candidate_title(cv_text)
    about = extract_about(cv_text)
    emp_type = extract_employment_type(cv_text)

    return {
        "candidate_email": email,
        "full_name": name,
        "candidate_title": title,
        "about": about,
        "location": location,
        "preferred_employment_type": emp_type,
        "yoe": yoe,
        "seniority": seniority,
        "created_at": created_at
    }