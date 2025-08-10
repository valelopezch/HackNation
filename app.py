import streamlit as st
import pandas as pd
import os 

from data_utils import (
    load_users, load_recruiters, load_jobs, load_candidates, load_applications,
    authenticate_user, save_application, apps_for_job, apps_for_candidate,
    candidate_skills_map, global_skill_pool_from_candidates
)
from embed import HybridMatcher, build_candidate_text
from validate import build_quiz, grade_quiz, load_mcq_dataset, default_answers_skeleton


st.set_page_config(page_title="TalentAI", page_icon="🧠", layout="wide")

# -----------------------------
# Session & constants
# -----------------------------
DEFAULTS = {
    "auth_ok": False,
    "user_email": "",
    "role": "",              # "recruiter" or "candidate"
    "full_name": "",
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# Cached data & resources
# -----------------------------

@st.cache_data(show_spinner=False)
def _load_all_data():
    users = load_users()
    recruiters = load_recruiters()
    jobs = load_jobs()
    cands = load_candidates()
    apps = load_applications()
    return users, recruiters, jobs, cands, apps

@st.cache_resource(hash_funcs={pd.DataFrame: lambda _: None})
def _matcher(jobs_df, skills_map):
    return HybridMatcher(jobs_df, skills_map=skills_map)

# -----------------------------
# UI helpers
# -----------------------------

def logout():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

def header_nav():
    cols = st.columns([6,1.5,2,1.2])
    with cols[0]:
        st.subheader("🧠 TalentAI")
    if st.session_state.get("auth_ok"):
        with cols[1]:
            st.caption(f"Role: **{st.session_state['role'].capitalize()}**")
        with cols[2]:
            st.caption(st.session_state["user_email"])
        with cols[3]:
            if st.button("Logout", use_container_width=True):
                logout()
    st.markdown("---")

def login_view():
    st.title("Sign in")
    st.caption("Use any account present in **data/users.csv**.")
    with st.form("login"):
        u = st.text_input("Email")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        auth = authenticate_user(u, p)
        if auth:
            st.session_state.auth_ok = True
            st.session_state.user_email = auth["email"]
            st.session_state.role = auth["role"]
            st.session_state.full_name = auth["full_name"]
            st.success(f"Welcome, {st.session_state.full_name or st.session_state.user_email}!")
            st.rerun()
        else:
            st.error("Invalid credentials")

# -----------------------------
# Recruiter views
# -----------------------------

def recruiter_home(jobs: pd.DataFrame, cands: pd.DataFrame):
    st.header("Recruiter Dashboard")
    my_jobs = jobs[jobs["posted_by"].str.lower() == st.session_state.user_email.lower()] \
              if "posted_by" in jobs.columns else jobs

    st.subheader("My Jobs")
    if my_jobs.empty:
        st.info("No jobs found for your account.")
        return

    st.dataframe(
        my_jobs[["job_id","job_title","topic","site (remote country)","seniority","yoe","employment_type"]],
        use_container_width=True, height=280
    )

    # Select a job
    job_id = st.selectbox("Select a job to analyze", my_jobs["job_id"].tolist())
    job_row = my_jobs[my_jobs["job_id"] == job_id].iloc[0]

    # Build candidate skills map from candidates.csv (since skills.csv was removed)
    skills_map = candidate_skills_map(cands)
    matcher = _matcher(jobs, skills_map)
    global_rank = matcher.score_job_vs_candidates(job_row, cands, skills_map).head(20)

    st.subheader("Top Matches (All Candidates)")
    st.dataframe(global_rank, use_container_width=True, height=360)

    # Applicants-only leaderboard
    st.subheader("Top Matches (Applicants Only)")
    apps = apps_for_job(job_id)
    if apps.empty:
        st.info("No applicants yet.")
    else:
        appl_emails = apps["candidate_email"].unique().tolist()
        appl_cands  = cands[cands["candidate_email"].isin(appl_emails)]

        skills_map = candidate_skills_map(cands)
        rank_appl = matcher.score_job_vs_candidates(job_row, appl_cands, skills_map)

        # Renombrar para evitar colisión
        rank_appl = rank_appl[["candidate_email", "score_match"]].rename(
            columns={"score_match": "score_match_rank"}
        )

        merged = apps.merge(rank_appl, on="candidate_email", how="left")

        # Preferimos el score calculado al vuelo; fallback al guardado en apps
        merged["score_match_final"] = merged["score_match_rank"].combine_first(
            merged["score_match"]
        )

        merged = merged.sort_values(
            ["score_match_final", "score_validation"], ascending=False
        )

        st.dataframe(
            merged[["candidate_email", "score_validation", "score_match_final", "status", "created_at"]]
                .rename(columns={"score_match_final": "score_match"}),
            use_container_width=True, height=300
        )

# -----------------------------
# Candidate views
# -----------------------------

def candidate_home(jobs: pd.DataFrame, cands: pd.DataFrame):
    st.header("Candidate")

    # ---------- Load (or reuse) MCQ dataset ----------
    if "mcq_df" not in st.session_state:
        # Look in data/ and project root
        DATA_DIR = os.path.join(os.getcwd(), "data") 
        mcq_paths = [
            os.path.join(DATA_DIR, "interview_questions_mcq.csv"),
            "interview_questions_mcq.csv",
        ]
        mcq_df = pd.DataFrame(columns=["topic", "question", "options", "correct_answer", "options_parsed"])
        for p in mcq_paths:
            try:
                if os.path.exists(p):
                    mcq_df = load_mcq_dataset(p)
                    st.session_state["mcq_path"] = p
                    break
            except Exception as e:
                pass
        st.session_state["mcq_df"] = mcq_df

    mcq_df = st.session_state["mcq_df"]
    if mcq_df.empty:
        st.warning("No MCQ questions loaded. Please add data/interview_questions_mcq.csv (or project root).")

    # ---------- Candidate profile ----------
    me = cands[cands["candidate_email"].str.lower() == st.session_state.user_email.lower()]
    if me.empty:
        st.warning("We didn’t find your profile in candidates.csv. A minimal profile will be used.")
        me_row = pd.Series({
            "candidate_email": st.session_state.user_email,
            "full_name": st.session_state.full_name,
            "candidate_title": "",
            "about": "",
            "skills": "",
            "yoe": 0,
            "seniority": "",
            "location": "",
            "preferred_employment_type": ""
        })
    else:
        me_row = me.iloc[0]

    skills_text = str(me_row.get("skills",""))
    cand_skills = set(s.strip() for s in skills_text.split(",") if s.strip())
    cand_text = build_candidate_text(me_row, skills_text)

    # ---------- Matching ----------
    skills_map = candidate_skills_map(cands)
    matcher = _matcher(jobs, skills_map)
    rank = matcher.score_candidate_vs_jobs(cand_text, me_row, cand_skills).head(20)

    st.subheader("Top Matching Jobs")
    st.dataframe(rank[["job_id","job_title","topic","score_match"]], use_container_width=True, height=360)

    # Keep per-job quiz/answers in session
    if "quizzes" not in st.session_state:
        st.session_state["quizzes"] = {}
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_results" not in st.session_state:
        st.session_state["quiz_results"] = {}

    # ---------- Apply + validation (MCQ-only quiz by topic) ----------
    with st.expander("Apply to a job"):
        options = rank["job_id"].tolist()
        chosen = st.selectbox("Choose a job to apply", options) if options else None

        if chosen:
            job_row = jobs[jobs["job_id"] == chosen].iloc[0]
            st.write(f"*{job_row['job_title']}* — {job_row.get('topic','')}")

            # Build the quiz once per job_id (MCQ dataset only)
            if chosen not in st.session_state["quizzes"]:
                quiz = build_quiz(
                    job_row,
                    global_skill_pool=[],     # not used by builder
                    mcq_df=mcq_df,            # REQUIRED
                )
                st.session_state["quizzes"][chosen] = quiz
                st.session_state["quiz_answers"][chosen] = default_answers_skeleton(quiz)

            quiz = st.session_state["quizzes"][chosen]
            answers = st.session_state["quiz_answers"][chosen]

            if not quiz:
                st.warning("No questions available for this topic. Check that the job topic matches the CSV topics.")
            else:

                with st.form(key=f"quiz_form_{chosen}", clear_on_submit=False):
                    for i, q in enumerate(quiz):
                        st.markdown(f"*Q{i+1}. {q['question']}*")
                        # Single-choice: store selected index
                        opts = list(enumerate(q["options"]))
                        default_idx = answers[i] if isinstance(answers[i], int) and 0 <= answers[i] < len(opts) else 0
                        sel = st.radio(
                            label="",
                            options=opts,
                            format_func=lambda p: p[1],
                            index=default_idx,
                            key=f"{chosen}_q{i}_single",
                        )
                        answers[i] = sel[0]
                        st.write("---")

                    submitted = st.form_submit_button("Submit validation", use_container_width=True)

                # Persist current answers
                st.session_state["quiz_answers"][chosen] = answers

                if submitted:
                    res = grade_quiz(answers, quiz)
                    st.session_state["quiz_results"][chosen] = res
                    st.success(f"Validation score: {res['score_raw']}/{res['score_max']} ({res['score_pct']}%)")

                    # Save application with validation + match score
                    mrow = rank[rank["job_id"] == chosen].iloc[0]
                    saved = save_application(
                        chosen,
                        st.session_state.user_email,
                        res["score_pct"],
                        round(float(mrow["score_match"]), 4),
                    )
                    st.balloons()
                    st.info("Successfully applied! You can review your applications below.")

    # ---------- My applications ----------
    st.subheader("My Applications")
    apps = apps_for_candidate(st.session_state.user_email)
    if apps.empty:
        st.caption("No applications yet.")
    else:
        st.dataframe(apps, use_container_width=True, height=260)

# -----------------------------
# Router
# -----------------------------

def home_page():
    header_nav()
    _users, _recruiters, jobs, cands, _apps = _load_all_data()

    _split = lambda s: tuple(x.strip() for x in str(s).split(",") if x.strip())
    cands["skills"] = cands["skills"].apply(_split)
    jobs["skills"]  = jobs["Skills/Tech-stack required"].apply(_split)

    skills_map = candidate_skills_map(cands)
    matcher = _matcher(jobs, skills_map)


    if not st.session_state.get("auth_ok"):
        login_view()
        return

    if st.session_state.role == "recruiter":
        recruiter_home(jobs, cands)
    elif st.session_state.role == "candidate":
        candidate_home(jobs, cands)
    else:
        st.warning("Unknown role. Please logout and sign in again.")

if __name__ == "__main__":
    home_page()