import streamlit as st
import pandas as pd
import os 
import re

from data_utils import (
    load_users, load_recruiters, load_jobs, load_candidates, load_applications,
    authenticate_user, save_application, apps_for_job, apps_for_candidate,
    candidate_skills_map, global_skill_pool_from_candidates, upsert_recruiter_profile, 
    create_user, upsert_candidate_profile, extract_cv_fields, get_signatures
)
from embed import DualMatcher, build_candidate_text, read_manifest
from validate import build_quiz, grade_quiz, load_mcq_dataset, default_answers_skeleton

import hashlib, json
from PyPDF2 import PdfReader
import tempfile

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

def df_sha256(df: pd.DataFrame) -> str:
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def cfg_sha256(model_name="paraphrase-multilingual-MiniLM-L12-v2", tfidf_min_df=1, tfidf_ngram=(1,2)) -> str:
    cfg = {"model_name": model_name, "tfidf_min_df": tfidf_min_df, "tfidf_ngram": tfidf_ngram}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()

@st.cache_data(show_spinner=False)
def _load_all_data(users_sig, recs_sig, jobs_sig, cands_sig, apps_sig):
    users = load_users()
    recruiters = load_recruiters()
    jobs = load_jobs()
    cands = load_candidates()
    apps = load_applications()
    return users, recruiters, jobs, cands, apps

@st.cache_resource(show_spinner=False)
def _matcher(jobs_sig: str, cands_sig: str, cfg_sig: str, jobs_df: pd.DataFrame, cands_df: pd.DataFrame, skills_map: dict):
    return DualMatcher(jobs_df, cands_df, skills_map=skills_map, store_root="./vector_store")

# -----------------------------
# UI helpers
# -----------------------------

def clean_skill_string(skill):
    """Clean skill string by removing leading/trailing parentheses, quotes, and whitespace"""
    if not skill:
        return ""
    # Remove leading/trailing parentheses, quotes, and whitespace
    cleaned = skill.strip().strip("()").strip("'").strip('"').strip()
    return cleaned

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

# def login_view():
#     st.title("Sign in")
#     st.caption("Use any account present in **data/users.csv**.")
#     with st.form("login"):
#         u = st.text_input("Email")
#         p = st.text_input("Password", type="password")
#         submitted = st.form_submit_button("Sign in")
#     if submitted:
#         auth = authenticate_user(u, p)
#         if auth:
#             st.session_state.auth_ok = True
#             st.session_state.user_email = auth["email"]
#             st.session_state.role = auth["role"]
#             st.session_state.full_name = auth["full_name"]
#             st.success(f"Welcome, {st.session_state.full_name or st.session_state.user_email}!")
#             st.rerun()
#         else:
#             st.error("Invalid credentials")

def is_valid_password(password: str) -> bool:
    """Check if password > 8 chars and contains at least one special char."""
    # return len(password) > 8 and bool(re.search(r"[^A-Za-z0-9]", password))
    return True


def login_view():
    st.title("Welcome to TalentAI")
    tabs = st.tabs(["Sign in", "Sign up"])

    # --- SIGN IN ---
    with tabs[0]:
        with st.form("login"):
            u = st.text_input("Email")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            if not is_valid_password(p):
                st.error("Invalid credentials")
            else:
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

    # --- SIGN UP ---
    with tabs[1]:
        st.caption("Create a new account")

        st.markdown("### Initial details")
        colA, colB = st.columns(2)
        with colA:
            email = st.text_input("Email *", key="su_email")
            full_name = st.text_input("Full name *", key="su_full_name")
            password = st.text_input("Password *", type="password", key="su_password")
        with colB:
            password2 = st.text_input("Confirm password *", type="password", key="su_password2")
            location = st.text_input("Location", key="su_location")
            role = st.selectbox("Role *", ["", "candidate", "recruiter"], index=0, key="su_role")

        cand_title = about = skills = preferred_type = seniority = ""
        yoe = "0"
        company_name = company_site = company_bio = ""
        if role == "candidate":
            st.markdown("### Complete your profile")
            uploaded = st.file_uploader("Upload your CV (PDF only)", type=["pdf"], key="cv_upload")
            parsed = {}
            if uploaded is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read()); tmp_path = tmp.name
                try:
                    text = ""
                    reader = PdfReader(tmp_path)
                    for p in reader.pages:
                        text += (p.extract_text() or "") + "\n"
                finally:
                    os.remove(tmp_path)
                parsed = extract_cv_fields(text, full_name, email) or {}

            c1, c2 = st.columns(2)
            with c1:
                cand_title   = st.text_input("Candidate title", value=parsed.get("candidate_title",""))
                yoe          = st.text_input("Years of experience (YOE)", value=str(parsed.get("yoe","0")))
                seniority    = st.text_input("Seniority", value=parsed.get("seniority",""))
            with c2:
                preferred_type = st.text_input("Preferred employment type", value=parsed.get("preferred_employment_type",""))
                skills         = st.text_input("Skills (comma-separated)")
            about = st.text_area("About", value=parsed.get("about",""), height=90)

        elif role == "recruiter":
            st.markdown("### Recruiter details")
            r1, r2 = st.columns(2)
            with r1:
                company_name = st.text_input("Company name")
                company_site = st.text_input("Company site (URL)")
            with r2:
                company_bio = st.text_area("Company bio", height=90)

        # Final submit
        st.markdown("---")
        with st.form("signup_submit"):
            agree = st.checkbox("I confirm the information is correct", value=True)
            submit_up = st.form_submit_button("Create account")

        if submit_up:
            if not email or not full_name or not password or not password2 or not role:
                st.error("Please fill all required fields.")
                return
            if not is_valid_password(password):
                st.error("Invalid credentials")
                return
            if password != password2:
                st.error("Passwords do not match.")
                return

            try:
                created_user = create_user(
                    email=email, password=password, role=role,
                    full_name=full_name, location=location
                )
                if role == "candidate":
                    upsert_candidate_profile(
                        candidate_email=email,
                        full_name=full_name,
                        candidate_title=cand_title,
                        about=about,
                        location=location,
                        preferred_employment_type=preferred_type,
                        yoe=yoe,
                        seniority=seniority,
                        skills=skills
                    )
                else:
                    upsert_recruiter_profile(
                        email=email,
                        company_name=company_name,
                        company_site=company_site,
                        bio=company_bio,
                        posted_jobs=0
                    )

                st.success("Account created! Logging you in…")
                st.session_state.auth_ok   = True
                st.session_state.user_email = created_user["email"]
                st.session_state.role       = created_user["role"]
                st.session_state.full_name  = created_user["full_name"]
                st.cache_data.clear(); st.cache_resource.clear()
                st.rerun()
            except ValueError as ve:
                st.error(str(ve))
            except Exception as e:
                st.exception(e)
# -----------------------------
# Recruiter views
# -----------------------------

# def recruiter_home(jobs: pd.DataFrame, cands: pd.DataFrame, matcher):
#     st.header("Recruiter Dashboard")
#     my_jobs = jobs[jobs["posted_by"].str.lower() == st.session_state.user_email.lower()] \
#               if "posted_by" in jobs.columns else jobs

#     st.subheader("My Jobs")
#     if my_jobs.empty:
#         st.info("No jobs found for your account.")
#         return

#     st.dataframe(
#         my_jobs[["job_id","job_title","topic","site (remote country)","seniority","yoe","employment_type"]],
#         use_container_width=True, height=280
#     )

#     # Select a job
#     job_id = st.selectbox("Select a job to analyze", my_jobs["job_id"].tolist())
#     job_row = my_jobs[my_jobs["job_id"] == job_id].iloc[0]

#     # Build candidate skills map from candidates.csv (since skills.csv was removed)
#     skills_map = candidate_skills_map(cands)
#     # jobs_sig = df_sha256(jobs)
#     # cfg_sig  = cfg_sha256()
#     # matcher = _matcher(jobs, skills_map, jobs_sig, cfg_sig)
#     global_rank = matcher.score_job_vs_candidates(job_row, cands, skills_map).head(20)

#     st.subheader("Top Matches (All Candidates)")
#     st.dataframe(global_rank, use_container_width=True, height=360)

#     # Applicants-only leaderboard
#     st.subheader("Top Matches (Applicants Only)")
#     apps = apps_for_job(job_id)
#     if apps.empty:
#         st.info("No applicants yet.")
#     else:
#         appl_emails = apps["candidate_email"].unique().tolist()
#         appl_cands  = cands[cands["candidate_email"].isin(appl_emails)]

#         skills_map = candidate_skills_map(cands)
#         rank_appl = matcher.score_job_vs_candidates(job_row, appl_cands, skills_map)

#         # Renombrar para evitar colisión
#         rank_appl = rank_appl[["candidate_email", "score_match"]].rename(
#             columns={"score_match": "score_match_rank"}
#         )

#         merged = apps.merge(rank_appl, on="candidate_email", how="left")

#         # Preferimos el score calculado al vuelo; fallback al guardado en apps
#         merged["score_match_final"] = merged["score_match_rank"].combine_first(
#             merged["score_match"]
#         )

#         merged = merged.sort_values(
#             ["score_match_final", "score_validation"], ascending=False
#         )

#         st.dataframe(
#             merged[["candidate_email", "score_validation", "score_match_final", "status", "created_at"]]
#                 .rename(columns={"score_match_final": "score_match"}),
#             use_container_width=True, height=300
#         )

def recruiter_home(jobs: pd.DataFrame, cands: pd.DataFrame, matcher):
    st.header("Recruiter Dashboard")

    # --- New: Post a job ---
    with st.expander("➕ Post a new job", expanded=False):
        with st.form("new_job_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                topic = st.text_input("Topic (e.g., Computer Vision, MLE, LLMs)", "")
                job_title = st.text_input("Job title *", "")
                site_remote_country = st.text_input("Site (remote country)", "")
                seniority = st.text_input("Seniority (e.g., Junior, Mid, Senior)", "")
                yoe = st.text_input("Years of experience (required)", "")
                employment_type = st.text_input("Employment type (Full-time / Contract / ...)", "")
            with c2:
                skills_tech = st.text_area("Skills / Tech stack (comma-separated) *", height=90)
                educational_reqs = st.text_area("Educational requirements", height=90)
                perks_benefits = st.text_area("Perks / Benefits", height=90)
                tasks = st.text_area("Main tasks", height=120)
            extra_info = st.text_area("Extra info", height=80)

            submitted_new_job = st.form_submit_button("Create job")

        if submitted_new_job:
            if not job_title or not skills_tech:
                st.error("Please fill at least Job title and Skills/Tech stack.")
            else:
                try:
                    from data_utils import create_job
                    created = create_job(
                        posted_by=st.session_state.user_email,
                        topic=topic,
                        job_title=job_title,
                        site_remote_country=site_remote_country,
                        tasks=tasks,
                        perks_benefits=perks_benefits,
                        skills_tech=skills_tech,
                        educational_reqs=educational_reqs,
                        seniority=seniority,
                        yoe=yoe,
                        employment_type=employment_type,
                        extra_info=extra_info
                    )
                    st.success(f"Job created: {created['job_title']} ({created['job_id']})")
                    # IMPORTANT: refresh caches so vector stores rebuild and include this job
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.rerun()
                except Exception as e:
                    st.exception(e)

    # --- Existing: My Jobs + matching ---
    my_jobs = jobs[jobs["posted_by"].str.lower() == st.session_state.user_email.lower()] \
              if "posted_by" in jobs.columns else jobs

    st.subheader("💼 My Jobs")
    if my_jobs.empty:
        st.info("No jobs found for your account yet. Use **Post a new job** above.")
        return

    st.dataframe(
        my_jobs[["job_id","job_title","topic","site (remote country)","seniority","yoe","employment_type"]],
        use_container_width=True, height=280
    )

    # Select a job
    job_id = st.selectbox("Select a job to analyze", my_jobs["job_id"].tolist())
    job_row = my_jobs[my_jobs["job_id"] == job_id].iloc[0]

    # Build candidate skills map
    skills_map = candidate_skills_map(cands)
    global_rank = matcher.score_job_vs_candidates(job_row, cands, skills_map).head(20)

    st.subheader("🏆 Top Matches (All Candidates)")
    st.dataframe(global_rank, use_container_width=True, height=360)

    # Applicants-only leaderboard (unchanged)
    st.subheader("🎯 Top Matches (Applicants Only)")
    apps = apps_for_job(job_id)
    if apps.empty:
        st.info("No applicants yet.")
    else:
        appl_emails = apps["candidate_email"].unique().tolist()
        appl_cands  = cands[cands["candidate_email"].isin(appl_emails)]

        skills_map = candidate_skills_map(cands)
        rank_appl = matcher.score_job_vs_candidates(job_row, appl_cands, skills_map)
        rank_appl = rank_appl[["candidate_email", "score_match"]].rename(columns={"score_match": "score_match_rank"})
        merged = apps.merge(rank_appl, on="candidate_email", how="left")
        merged["score_match_final"] = merged["score_match_rank"].combine_first(merged["score_match"])
        merged = merged.sort_values(["score_match_final", "score_validation"], ascending=False)

        # Enhanced applicants dashboard
        st.caption(f"Showing {len(merged)} applicant(s) for this position")
        
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📊 Dashboard View", "📋 Table View"])
        
        with tab1:
            # Dashboard view with detailed applicant information
            for idx, app in merged.iterrows():
                # Get candidate details for this application
                candidate_details = cands[cands["candidate_email"] == app["candidate_email"]]
                if candidate_details.empty:
                    continue
                    
                candidate = candidate_details.iloc[0]
                
                # Create an expander for each applicant
                with st.expander(f"👤 {candidate.get('full_name', 'Unknown')} - {candidate.get('candidate_title', 'N/A')}", expanded=False):
                    # Header with key information
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Email:** {candidate.get('candidate_email', 'N/A')}")
                        st.markdown(f"**Location:** {candidate.get('location', 'N/A')}")
                        st.markdown(f"**Preferred Employment:** {candidate.get('preferred_employment_type', 'N/A')}")
                    
                    with col2:
                        # Match score analysis
                        match_score = app.get('score_match_final', 0)
                        if match_score:
                            match_pct = min(100, max(0, float(match_score) * 100))
                            st.metric("Match Score", f"{match_pct:.1f}%")
                            
                            # Color-coded match indicator
                            if match_pct >= 70:
                                st.success("🎯 Excellent Match")
                            elif match_pct >= 50:
                                st.info("✅ Good Match")
                            elif match_pct >= 30:
                                st.warning("⚠️ Moderate Match")
                            else:
                                st.error("❌ Low Match")
                        else:
                            st.info("Match score not available")
                    
                    with col3:
                        # Validation score
                        val_score = app.get('score_validation', 0)
                        if val_score:
                            st.metric("Validation Score", f"{val_score:.0f}%")
                            if val_score >= 80:
                                st.success("🎓 Excellent")
                            elif val_score >= 60:
                                st.info("✅ Good")
                            elif val_score >= 40:
                                st.warning("⚠️ Fair")
                            else:
                                st.error("❌ Needs Improvement")
                        else:
                            st.info("Validation not completed")
                    
                    st.markdown("---")
                    
                    # Candidate details section
                    st.subheader("👤 Candidate Profile")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Seniority Level:** {candidate.get('seniority', 'N/A')}")
                        st.markdown(f"**Years of Experience:** {candidate.get('yoe', 'N/A')}")
                        st.markdown(f"**Application Status:** {app.get('status', 'N/A').title()}")
                        st.markdown(f"**Applied:** {app.get('created_at', 'N/A')}")
                        
                        # Skills
                        if candidate.get('skills'):
                            st.markdown("**Skills:**")
                            skills_list = [clean_skill_string(skill) for skill in str(candidate['skills']).split(',') if skill.strip()]
                            for skill in skills_list[:10]:  # Show first 10 skills
                                st.markdown(f"• {skill}")
                            if len(skills_list) > 10:
                                st.caption(f"... and {len(skills_list) - 10} more skills")
                    
                    with col2:
                        st.markdown(f"**About:** {candidate.get('about', 'N/A')}")
                        
                        # Application performance summary
                        st.markdown("**📊 Application Summary:**")
                        if match_score and val_score:
                            overall_score = (match_pct + val_score) / 2
                            st.metric("Overall Score", f"{overall_score:.1f}%")
                            
                            if overall_score >= 75:
                                st.success("🌟 **Top Candidate**")
                            elif overall_score >= 60:
                                st.info("✅ **Strong Candidate**")
                            elif overall_score >= 45:
                                st.warning("⚠️ **Consider**")
                            else:
                                st.error("❌ **May Need More Review**")
                        else:
                            st.info("Complete scoring not available")
                    
                    # About section (if available)
                    if candidate.get('about') and len(str(candidate['about'])) > 50:
                        st.markdown("**📝 About Candidate:**")
                        st.info(candidate['about'])
                    
                    # Why this candidate is a good match analysis
                    st.markdown("---")
                    st.subheader("🔍 Why This Candidate is a Good Match (or Not)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if match_score:
                            st.markdown("**Match Analysis:**")
                            if match_pct >= 70:
                                st.success(f"🎯 **Excellent Match!** This candidate's profile aligns very well with the position requirements. The {match_pct:.1f}% match score indicates strong compatibility in skills, experience, and qualifications.")
                            elif match_pct >= 50:
                                st.info(f"✅ **Good Match** - The candidate shows good alignment with this role. The {match_pct:.1f}% match score suggests they meet most key requirements.")
                            elif match_pct >= 30:
                                st.warning(f"⚠️ **Moderate Match** - There's some alignment, but the candidate may need to highlight specific skills or experience to improve their fit.")
                            else:
                                st.error(f"❌ **Low Match** - This candidate may not be the best fit for this position based on their current profile.")
                        else:
                            st.info("Match analysis not available for this application.")
                    
                    with col2:
                        if val_score:
                            st.markdown("**Validation Performance:**")
                            if val_score >= 80:
                                st.success(f"🎓 **Excellent Performance** - The candidate scored {val_score:.0f}% on the validation quiz, demonstrating strong knowledge in this domain.")
                            elif val_score >= 60:
                                st.info(f"✅ **Good Performance** - Their {val_score:.0f}% score shows solid understanding of the required concepts.")
                            elif val_score >= 40:
                                st.warning(f"⚠️ **Fair Performance** - Their {val_score:.0f}% score indicates some knowledge gaps that could be addressed.")
                            else:
                                st.error(f"❌ **Needs Improvement** - Their {val_score:.0f}% score suggests focusing on building knowledge in this area.")
                        else:
                            st.info("Validation quiz not completed for this application.")
                    
                    # Job-specific insights
                    st.markdown("---")
                    st.subheader("🎯 Position-Specific Insights")
                    
                    # Compare candidate skills with job requirements
                    if candidate.get('skills') and job_row.get('Skills/Tech-stack required'):
                        candidate_skills = set(s.strip().lower() for s in str(candidate['skills']).split(',') if s.strip())
                        job_skills = set(s.strip().lower() for s in str(job_row['Skills/Tech-stack required']).split(',') if s.strip())
                        
                        matching_skills = candidate_skills.intersection(job_skills)
                        missing_skills = job_skills - candidate_skills
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if matching_skills:
                                st.markdown("**✅ Matching Skills:**")
                                for skill in sorted(list(matching_skills))[:8]:
                                    st.markdown(f"• {skill}")
                                if len(matching_skills) > 8:
                                    st.caption(f"... and {len(matching_skills) - 8} more")
                            else:
                                st.warning("No matching skills found")
                        
                        with col2:
                            if missing_skills:
                                st.markdown("**❌ Missing Skills:**")
                                for skill in sorted(list(missing_skills))[:8]:
                                    st.markdown(f"• {skill}")
                                if len(missing_skills) > 8:
                                    st.caption(f"... and {len(missing_skills) - 8} more")
                            else:
                                st.success("All required skills are covered!")
        
        with tab2:
            # Original table view for quick reference
            st.dataframe(
                merged[["candidate_email", "score_validation", "score_match_final", "status", "created_at"]]
                    .rename(columns={"score_match_final": "score_match"}),
                use_container_width=True, height=400
            )

# -----------------------------
# Candidate views
# -----------------------------

def candidate_home(jobs: pd.DataFrame, cands: pd.DataFrame, matcher):
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

    # ---------- Profile editing ----------
    if st.button("✏️ Edit my profile"):
        with st.form("edit_profile_form"):
            full_name = st.text_input("Full name", value=me_row.get("full_name", ""))
            cand_title = st.text_input("Candidate title", value=me_row.get("candidate_title", ""))
            about = st.text_area("About", value=me_row.get("about", ""))
            skills = st.text_input("Skills (comma-separated)", value=me_row.get("skills", ""))
            yoe = st.text_input("Years of experience (YOE)", value=str(me_row.get("yoe", "0")))
            seniority = st.text_input("Seniority", value=me_row.get("seniority", ""))
            location = st.text_input("Location", value=me_row.get("location", ""))
            preferred_type = st.text_input("Preferred employment type", value=me_row.get("preferred_employment_type", ""))

            submitted = st.form_submit_button("Save changes")
            if submitted:
                upsert_candidate_profile(
                    candidate_email=st.session_state.user_email,
                    full_name=full_name,
                    candidate_title=cand_title,
                    about=about,
                    location=location,
                    preferred_employment_type=preferred_type,
                    yoe=yoe,
                    seniority=seniority,
                    skills=skills
                )
                st.success("Profile updated!")
                st.rerun()

    # ---------- Matching ----------
    # skills_map = candidate_skills_map(cands)
    # jobs_sig = df_sha256(jobs)
    # cfg_sig  = cfg_sha256()
    # matcher = _matcher(jobs, skills_map, jobs_sig, cfg_sig)
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
    with st.expander("🔍 Apply to a job"):
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
    st.subheader("📋 My Applications")
    apps = apps_for_candidate(st.session_state.user_email)
    if apps.empty:
        st.caption("No applications yet.")
    else:
        # Enhanced applications dashboard
        st.caption(f"Showing {len(apps)} application(s)")
        
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["📊 Dashboard View", "📋 Table View"])
        
        with tab1:
            # Dashboard view with detailed information
            for idx, app in apps.iterrows():
                # Get job details for this application
                job_details = jobs[jobs["job_id"] == app["job_id"]]
                if job_details.empty:
                    continue
                    
                job = job_details.iloc[0]
                
                # Create an expander for each application
                with st.expander(f"🎯 {job['job_title']} - {job.get('topic', 'N/A')}", expanded=False):
                    # Header with key information
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**Company:** {job.get('posted_by', 'N/A')}")
                        st.markdown(f"**Location:** {job.get('site (remote country)', 'N/A')}")
                        st.markdown(f"**Employment Type:** {job.get('employment_type', 'N/A')}")
                    
                    with col2:
                        # Match score analysis
                        match_score = app.get('score_match', 0)
                        if match_score:
                            match_pct = min(100, max(0, float(match_score) * 100))
                            st.metric("Match Score", f"{match_pct:.1f}%")
                            
                            # Color-coded match indicator
                            if match_pct >= 70:
                                st.success("🎯 Excellent Match")
                            elif match_pct >= 50:
                                st.info("✅ Good Match")
                            elif match_pct >= 30:
                                st.warning("⚠️ Moderate Match")
                            else:
                                st.error("❌ Low Match")
                        else:
                            st.info("Match score not available")
                    
                    with col3:
                        # Validation score
                        val_score = app.get('score_validation', 0)
                        if val_score:
                            st.metric("Validation Score", f"{val_score:.0f}%")
                            if val_score >= 80:
                                st.success("🎓 Excellent")
                            elif val_score >= 60:
                                st.info("✅ Good")
                            elif val_score >= 40:
                                st.warning("⚠️ Fair")
                            else:
                                st.error("❌ Needs Improvement")
                        else:
                            st.info("Validation not completed")
                    
                    st.markdown("---")
                    
                    # Job details section
                    st.subheader("📋 Job Details")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Seniority Level:** {job.get('seniority', 'N/A')}")
                        st.markdown(f"**Years of Experience:** {job.get('yoe', 'N/A')}")
                        st.markdown(f"**Topic:** {job.get('topic', 'N/A')}")
                        
                        # Skills required
                        if job.get('Skills/Tech-stack required'):
                            st.markdown("**Skills Required:**")
                            skills_list = [clean_skill_string(skill) for skill in str(job['Skills/Tech-stack required']).split(',') if skill.strip()]
                            for skill in skills_list[:10]:  # Show first 10 skills
                                st.markdown(f"• {skill}")
                            if len(skills_list) > 10:
                                st.caption(f"... and {len(skills_list) - 10} more skills")
                    
                    with col2:
                        st.markdown(f"**Educational Requirements:** {job.get('Educational requirements', 'N/A')}")
                        st.markdown(f"**Application Status:** {app.get('status', 'N/A').title()}")
                        st.markdown(f"**Applied:** {app.get('created_at', 'N/A')}")
                        
                        # Perks/Benefits
                        if job.get('Perks/Benefits'):
                            st.markdown("**Perks & Benefits:**")
                            perks_list = [clean_skill_string(perk) for perk in str(job['Perks/Benefits']).split(',') if perk.strip()]
                            for perk in perks_list[:5]:  # Show first 5 perks
                                st.markdown(f"• {perk}")
                            if len(perks_list) > 5:
                                st.caption(f"... and {len(perks_list) - 5} more perks")
                    
                    # Tasks section
                    if job.get('tasks'):
                        st.markdown("**📝 Key Tasks:**")
                        tasks_list = [clean_skill_string(task) for task in str(job['tasks']).split('\n') if task.strip()]
                        for task in tasks_list[:8]:  # Show first 8 tasks
                            st.markdown(f"• {task}")
                        if len(tasks_list) > 8:
                            st.caption(f"... and {len(tasks_list) - 8} more tasks")
                    
                    # Extra info if available
                    if job.get('extra_info'):
                        st.markdown("**ℹ️ Additional Information:**")
                        st.info(job['extra_info'])
                    
                    # Why this is a good match analysis
                    st.markdown("---")
                    st.subheader("🔍 Why This Job is a Good Match (or Not)")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if match_score:
                            st.markdown("**Match Analysis:**")
                            if match_pct >= 70:
                                st.success(f"🎯 **Excellent Match!** Your profile aligns very well with this position. The {match_pct:.1f}% match score indicates strong compatibility in skills, experience, and requirements.")
                            elif match_pct >= 50:
                                st.info(f"✅ **Good Match** - Your profile shows good alignment with this role. The {match_pct:.1f}% match score suggests you meet most key requirements.")
                            elif match_pct >= 30:
                                st.warning(f"⚠️ **Moderate Match** - There's some alignment, but you may need to highlight specific skills or experience to improve your chances.")
                            else:
                                st.error(f"❌ **Low Match** - This position may not be the best fit based on current profile. Consider focusing on roles that better match your skills.")
                        else:
                            st.info("Match analysis not available for this application.")
                    
                    with col2:
                        if val_score:
                            st.markdown("**Validation Performance:**")
                            if val_score >= 80:
                                st.success(f"🎓 **Excellent Performance** - You scored {val_score:.0f}% on the validation quiz, demonstrating strong knowledge in this area.")
                            elif val_score >= 60:
                                st.info(f"✅ **Good Performance** - Your {val_score:.0f}% score shows solid understanding of the required concepts.")
                            elif val_score >= 40:
                                st.warning(f"⚠️ **Fair Performance** - Your {val_score:.0f}% score indicates some knowledge gaps that could be addressed.")
                            else:
                                st.error(f"❌ **Needs Improvement** - Your {val_score:.0f}% score suggests focusing on building knowledge in this domain.")
                        else:
                            st.info("Validation quiz not completed for this application.")
        
        with tab2:
            # Original table view for quick reference
            st.dataframe(apps, use_container_width=True, height=400)

# -----------------------------
# Router
# -----------------------------

def home_page():
    header_nav()
    sigs = get_signatures()
    _users, _recruiters, jobs, cands, _apps = _load_all_data(
        sigs["users"], sigs["recruiters"], sigs["jobs"], sigs["cands"], sigs["apps"]
    )

    # normaliza skills a secuencia (tu código actual)
    _split = lambda s: tuple(x.strip() for x in str(s).split(",") if x.strip())
    cands["skills"] = cands["skills"].apply(_split)
    jobs["skills"]  = jobs["Skills/Tech-stack required"].apply(_split)

    # matcher con firmas (clave de cache)
    skills_map = candidate_skills_map(cands)
    matcher = _matcher(sigs["jobs"], sigs["cands"], cfg_sha256(), jobs, cands, skills_map)

    with st.sidebar.expander("⚙️ Cache status", expanded=False):
        info = matcher.cache_status()
        st.write("Jobs store:", info["jobs_dir"])
        st.write("Cands store:", info["cands_dir"])
        st.write("Status:", info["status"])
        st.json(info["jobs_manifest"])
        st.json(info["cands_manifest"])

    if not st.session_state.get("auth_ok"):
        login_view()
        return

    if st.session_state.role == "recruiter":
        recruiter_home(jobs, cands, matcher)
    elif st.session_state.role == "candidate":
        candidate_home(jobs, cands, matcher)
    else:
        st.warning("Unknown role. Please logout and sign in again.")

if __name__ == "__main__":
    home_page()