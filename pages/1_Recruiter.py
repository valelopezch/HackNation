# pages/1_Recruiter.py
import streamlit as st, pandas as pd
from pages.index import CandidateIndex, hybrid_rank
from pages.embed import embed_texts

st.set_page_config(layout="wide")

def _get_state():
    return st.session_state

def _load_globals():
    # Access cache from app.py
    from app import load_data, load_index
    cands, jobs = load_data()
    idx = load_index(cands)
    return cands, jobs, idx

st.title("Recruiter – Search & Rank")
cands, jobs, idx: CandidateIndex = _load_globals()

colf, colw = st.columns([2,1])
with colf:
    job = st.selectbox("Pick a job", options=jobs["title"].tolist())
    sel = jobs[jobs["title"] == job].iloc[0]
    query_text = sel["description"]
    query_skills = set(sel["skills"])
with colw:
    region = st.selectbox("Region", options=["", *sorted(cands["region"].unique())])
    min_grad = st.number_input("Min graduation year", min_value=1990, max_value=2030, value=2018)
    avail = st.selectbox("Availability", options=["", *sorted(cands["availability"].unique())])
    w_sem = st.slider("Weight: semantic", 0.0, 1.0, 0.7, 0.05)
    w_skill = 1 - w_sem

shortlist_ids = idx.filter_ids(region or None, min_grad or None, avail or None)
raw = idx.search(query_text, k=100, shortlist_ids=shortlist_ids if shortlist_ids else None)
ranked = hybrid_rank(raw, idx.df, query_skills, w_sem=w_sem, w_skill=w_skill)

def _fmt_row(r):
    row = idx.df[idx.df["candidate_id"] == r["candidate_id"]].iloc[0]
    name = ("Candidate #" + str(r["candidate_id"])) if st.session_state.get("blind", True) else row["name"]
    return {
        "candidate_id": r["candidate_id"],
        "name": name,
        "score": round(r["score"], 3),
        "semantic": round(r["semantic"], 3),
        "skill_overlap": round(r["skill_overlap"], 3),
        "region": row["region"],
        "grad_year": row["grad_year"],
        "availability": row["availability"],
        "skills": ", ".join(row["skills"][:8])
    }

st.write("### Results")
tbl = pd.DataFrame([_fmt_row(r) for _, r in ranked.iterrows()])
st.dataframe(tbl, use_container_width=True, height=460)

with st.expander("Candidate detail"):
    cid = st.number_input("Candidate ID", value=int(tbl["candidate_id"].iloc[0]) if not tbl.empty else 0, step=1)
    if cid in idx.ids:
        row = idx.df[idx.df["candidate_id"] == cid].iloc[0]
        st.subheader(("Candidate #"+str(cid)) if st.session_state.get("blind", True) else row["name"])
        left, right = st.columns([2,1])
        with left:
            st.markdown("**Skills**: " + ", ".join(row["skills"]))
            st.markdown("**Profile text**")
            st.write(row["profile_text"][:1500] + ("..." if len(row["profile_text"])>1500 else ""))
        with right:
            st.metric("Grad year", row["grad_year"])
            st.metric("Region", row["region"])
            st.metric("Availability", row["availability"])
