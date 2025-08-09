import os, uuid
import pandas as pd
from slugify import slugify
from datetime import datetime

DATA_DIR = "data"
APP_FILE = os.path.join(DATA_DIR, "applications.csv")

def ensure_job_ids(df):
    if "job_id" not in df.columns:
        df = df.copy()
        df["job_id"] = df.apply(lambda r: slugify(f"{r.get('job_title','job')}-{r.name}"), axis=1)
    return df

def ensure_posted_by(df):
    if "posted_by" not in df.columns:
        df = df.copy()
        df["posted_by"] = "recruiter@demo"
    return df

def load_jobs():
    df = pd.read_csv(os.path.join(DATA_DIR,"jobs.csv"))
    df = ensure_job_ids(df)
    df = ensure_posted_by(df)
    return df

def load_candidates():
    df = pd.read_csv(os.path.join(DATA_DIR,"candidates.csv"))
    if "candidate_email" not in df.columns:
        # create a deterministic fake email from name + index
        df = df.copy()
        base = df.get("full_name","candidate")
        df["candidate_email"] = [f"cand{i}@demo" for i in range(len(df))]
    return df

def load_skills():
    df = pd.read_csv(os.path.join(DATA_DIR,"skills.csv"))
    # normalize to map email -> "skill1, skill2, ..."
    skill_map = {}
    if "skills" in df.columns:
        for _, r in df.iterrows():
            skill_map[r["candidate_email"]] = str(r["skills"])
    else:
        # assume rows candidate_email, skill
        tmp = df.groupby("candidate_email")["skill"].apply(lambda s: ", ".join(sorted(set(str(x) for x in s)))).reset_index()
        for _, r in tmp.iterrows():
            skill_map[r["candidate_email"]] = r["skill"]
    return df, skill_map

def list_global_skill_pool(skills_map: dict):
    all_text = ", ".join(skills_map.values()).split(",")
    return sorted(set(s.strip() for s in all_text if s.strip()))

def load_applications():
    if not os.path.exists(APP_FILE):
        pd.DataFrame(columns=["app_id","job_id","candidate_email","status","score_validation","score_match","created_at"]).to_csv(APP_FILE, index=False)
    return pd.read_csv(APP_FILE)

def save_application(job_id, candidate_email, score_validation, score_match, status="applied"):
    apps = load_applications()
    app = {
        "app_id": str(uuid.uuid4())[:8],
        "job_id": job_id,
        "candidate_email": candidate_email,
        "status": status,
        "score_validation": score_validation,
        "score_match": score_match,
        "created_at": datetime.utcnow().isoformat()
    }
    apps = pd.concat([apps, pd.DataFrame([app])], ignore_index=True)
    apps.to_csv(APP_FILE, index=False)
    return app

def apps_for_job(job_id):
    apps = load_applications()
    return apps[apps["job_id"] == job_id].copy()

def apps_for_candidate(email):
    apps = load_applications()
    return apps[apps["candidate_email"] == email].copy()
