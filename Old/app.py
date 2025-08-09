import streamlit as st
import pandas as pd
from pathlib import Path

# ----------------- App config -----------------
st.set_page_config(page_title="TalentAI", page_icon="🧠", layout="wide")

# ----------------- Session defaults -----------------
DEFAULTS = {
    "auth_ok": False,
    "user": "",
    "role": "",
    "blind": True,  # carried by recruiter page for bias-reduction
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ----------------- Demo users (MVP) -----------------
USERS = {
    "recruiter@demo": ("recruit123", "recruiter"),
    "cand_ana@demo": ("candidate123", "candidate"),
}

# ----------------- Shared cached loaders -----------------
@st.cache_data(show_spinner=False)
def load_data():
    # data/candidates.csv: candidate_id,name,region,grad_year,availability,skills(list-as-str),profile_text
    cand = pd.read_csv("data/candidates.csv")
    cand["skills"] = cand["skills"].apply(
        lambda s: eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    jobs = pd.read_csv("data/jobs.csv")  # job_id,title,region,skills(list-as-str),description
    jobs["skills"] = jobs["skills"].apply(
        lambda s: eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    return cand, jobs

# If you built a vector index (FAISS / sklearn) expose it here
@st.cache_resource(show_spinner=True)
def load_index(df):
    # Lazy import to keep app.py small
    from talentai.index import CandidateIndex
    return CandidateIndex(df)

# ----------------- Auth helpers -----------------
def logout():
    for k in DEFAULTS:
        st.session_state[k] = DEFAULTS[k]
    st.rerun()

def login_view():
    st.title("TalentAI – Sign in")
    with st.form("login"):
        u = st.text_input("Email / User")
        p = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        if u in USERS and USERS[u][0] == p:
            st.session_state.auth_ok = True
            st.session_state.user = u
            st.session_state.role = USERS[u][1]
            st.rerun()
        else:
            st.error("Invalid credentials")

# ----------------- Router -----------------
if not st.session_state.auth_ok:
    login_view()
    st.stop()

role = st.session_state.role
user = st.session_state.user

# Top bar
col1, col2, col3 = st.columns([6,2,1])
with col1:
    st.markdown(f"**Signed in as:** `{user}`  ·  **Role:** `{role}`")
with col2:
    st.session_state.blind = st.toggle("Blind mode", value=st.session_state.blind, help="Hide names/photos in rankings")
with col3:
    if st.button("Logout"):
        logout()

# Tabs by role (protected pages)
tabs = ("🏷️ Recruiter",) if role == "recruiter" else ("👤 Candidate",)
pages = ("pages._recruiter",) if role == "recruiter" else ("pages._candidate",)

st_tabs = st.tabs(list(tabs))
for title, tab, page_module in zip(tabs, st_tabs, pages):
    with tab:
        # Give pages access to shared caches via import app.load_*
        mod = __import__(page_module, fromlist=["render"])
        mod.render()
