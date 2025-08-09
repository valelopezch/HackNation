import streamlit as st
import pandas as pd

from data_utils import (
    load_jobs, load_candidates, load_skills, list_global_skill_pool,
    load_applications, save_application, apps_for_job, apps_for_candidate
)
from embed import TFIDFMatcher, build_candidate_text
from validate import build_quiz, grade_quiz

st.set_page_config(page_title="TalentAI", page_icon="🧠", layout="wide")

# --- Session keys
DEFAULTS = {
    "auth_ok": False,
    "user_email": "",
    "role": "",              # "recruiter" or "candidate"
    "full_name": "",
    "matcher_ready": False,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Demo users for Day 1 (swap for CSV/DB later)
USERS = {
    "recruiter@demo": {"password": "recruit123", "role": "recruiter", "full_name": "Rae Recruiter"},
    "cand_ana@demo":  {"password": "candidate123", "role": "candidate", "full_name": "Ana Candidate"},
}

@st.cache_data
def _load_all_data():
    jobs = load_jobs()
    cands = load_candidates()
    skills_df, skills_map = load_skills()
    global_skill_pool = list_global_skill_pool(skills_map)
    apps = load_applications()
    return jobs, cands, skills_df, skills_map, global_skill_pool, apps

@st.cache_resource
def _matcher(jobs_df):
    return TFIDFMatcher(jobs_df)

def logout():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()

def login_view():
    st.title("TalentAI – Sign in")
    with st.form("login"):
        u = st.text_input("Email / User")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if u in USERS and USERS[u]["password"] == p:
            st.session_state.auth_ok = True
            st.session_state.user_email = u
            st.session_state.role = USERS[u]["role"]
            st.session_state.full_name = USERS[u]["full_name"]
            st.success(f"Welcome, {st.session_state.full_name}!")
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

def header_nav():
    cols = st.columns([6,1,1,1])
    with cols[0]:
        st.subheader("🧠 TalentAI")
    if st.session_state.auth_ok:
        with cols[1]:
            st.write(f"**{st.session_state.role.capitalize()}**")
        with cols[2]:
            st.write(st.session_state.user_email)
        with cols[3]:
            if st.button("Logout", use_container_width=True):
                logout()
    st.markdown("---")

def recruiter_home(jobs, cands, skills_map):
    st.header("Recruiter Dashboard")

    # Show my jobs
    my = jobs[jobs["posted_by"] == st.session_state.user_email] if "posted_by" in jobs.columns else jobs
    st.subheader("My Jobs")
    st.dataframe(my[["job_id","job_title","topic","site (remote country)","seniority","yoe","employment_type"]], use_container_width=True)

    # Select job
    job_id = st.selectbox("Select a job to analyze", my["job_id"].tolist())
    job_row = my[my["job_id"] == job_id].iloc[0]

    matcher = _matcher(jobs)
    # Overall best candidates (global)
    global_rank = matcher.score_job_vs_candidates(job_row, cands, skills_map).head(15)

    st.subheader("Top Matches (All Candidates)")
    st.dataframe(global_rank, use_container_width=True, height=350)

    # Applicants-only leaderboard
    apps = apps_for_job(job_id)
    merged = apps.merge(global_rank, on="candidate_email", how="left").sort_values("score_match", ascending=False)
    st.subheader("Top Matches (Applicants Only)")
    if len(merged) == 0:
        st.info("No applicants yet.")
    else:
        st.dataframe(merged[["candidate_email","score_validation","score_match","created_at"]], use_container_width=True)

def candidate_home(jobs, cands, skills_map, global_skill_pool):
    st.header("Candidate")
    me = cands[cands["candidate_email"] == st.session_state.user_email]
    if me.empty:
        st.warning("We didn’t find your profile in candidates.csv. We’ll use a minimal profile.")
        me_row = pd.Series({"candidate_email": st.session_state.user_email, "full_name": st.session_state.full_name})
    else:
        me_row = me.iloc[0]

    # Build candidate text
    from embed import build_candidate_text
    skills_text = skills_map.get(st.session_state.user_email, "")
    cand_text = build_candidate_text(me_row, skills_text)

    matcher = _matcher(jobs)
    rank = matcher.score_candidate_vs_jobs(cand_text, me_row).head(20)

    st.subheader("Top Matching Jobs")
    st.dataframe(rank[["job_id","job_title","topic","score_match"]], use_container_width=True, height=350)

    # Apply & validate
    with st.expander("Apply to a job"):
        chosen = st.selectbox("Choose a job to apply", rank["job_id"].tolist())
        if chosen:
            job_row = jobs[jobs["job_id"] == chosen].iloc[0]
            st.write(f"**{job_row['job_title']}** — {job_row.get('topic','')}")
            # Build quiz
            quiz = build_quiz(job_row, global_skill_pool)
            answers = []
            for i, q in enumerate(quiz, 1):
                st.markdown(f"**Q{i}. {q['question']}**")
                if q.get("code"):
                    st.code(q["code"], language="python")
                if q["type"] == "single":
                    ans = st.radio("", options=list(range(len(q["options"]))), format_func=lambda k: q["options"][k], key=f"q{i}")
                else:
                    ans = st.multiselect("", options=list(range(len(q["options"]))), format_func=lambda k: q["options"][k], key=f"q{i}")
                answers.append(ans)

            if st.button("Submit validation"):
                res = grade_quiz(answers, quiz)
                st.success(f"Validation score: {res['score_raw']}/{res['score_max']} ({res['score_pct']}%)")

                # Match score for this job (already computed above)
                mrow = rank[rank["job_id"] == chosen].iloc[0]
                save_application(chosen, st.session_state.user_email, res["score_pct"], round(float(mrow["score_match"]), 4))

                st.balloons()
                st.info("Successfully applied! You can see your applications below.")

    st.subheader("My Applications")
    apps = apps_for_candidate(st.session_state.user_email)
    st.dataframe(apps, use_container_width=True)

def home_page():
    header_nav()
    jobs, cands, skills_df, skills_map, global_skill_pool, _apps = _load_all_data()

    # Keep credentials if already logged in
    if not st.session_state.auth_ok:
        login_view()
        return

    # Role router
    if st.session_state.role == "recruiter":
        recruiter_home(jobs, cands, skills_map)
    elif st.session_state.role == "candidate":
        candidate_home(jobs, cands, skills_map, global_skill_pool)
    else:
        st.warning("Unknown role. Please logout and sign in again.")

if __name__ == "__main__":
    home_page()
