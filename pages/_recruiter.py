# pages/_recruiter.py
import streamlit as st
import pandas as pd

def render():
    st.set_page_config(layout="wide")
    st.title("Recruiter – Search & Rank")

    # Bring shared cached objects from app
    from app import load_data, load_index
    cands, jobs = load_data()
    idx = load_index(cands)

    colf, colw = st.columns([2,1])
    with colf:
        job = st.selectbox("Pick a job", options=jobs["title"].tolist())
        sel = jobs[jobs["title"] == job].iloc[0]
        query_text = sel["description"]
        query_skills = set(sel["skills"])
        st.caption("Ranking = semantic × skill‑overlap; adjust weights →")
    with colw:
        region = st.selectbox("Region", options=["", *sorted(cands["region"].unique())])
        min_grad = st.number_input("Min graduation year", min_value=1990, max_value=2030, value=2018)
        avail = st.selectbox("Availability", options=["", *sorted(cands["availability"].unique())])
        w_sem = st.slider("Weight: semantic", 0.0, 1.0, 0.7, 0.05)
        w_skill = 1 - w_sem

    # Meta-filter → shortlist → semantic search → hybrid re-rank
    from pages.index import hybrid_rank
    shortlist_ids = idx.filter_ids(region or None, min_grad or None, avail or None)
    raw = idx.search(query_text, k=100, shortlist_ids=shortlist_ids if shortlist_ids else None)
    ranked = hybrid_rank(raw, idx.df, query_skills, w_sem=w_sem, w_skill=w_skill)

    def _fmt_row(r):
        row = idx.df[idx.df["candidate_id"] == r["candidate_id"]].iloc[0]
        show_name = not st.session_state.get("blind", True)
        name = ("Candidate #" + str(r["candidate_id"])) if not show_name else row["name"]
        # NOTE: above line flips name visibility; if you prefer to hide when blind=True, swap the conditional.
        name = row["name"] if not st.session_state.get("blind", True) else f"Candidate #{r['candidate_id']}"
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
        if not tbl.empty:
            default_id = int(tbl["candidate_id"].iloc[0])
        else:
            default_id = 0
        cid = st.number_input("Candidate ID", value=default_id, step=1)
        if cid in idx.ids:
            row = idx.df[idx.df["candidate_id"] == cid].iloc[0]
            title = ("Candidate #"+str(cid)) if st.session_state.get("blind", True) else row["name"]
            st.subheader(title)
            left, right = st.columns([2,1])
            with left:
                st.markdown("**Skills**: " + ", ".join(row["skills"]))
                st.markdown("**Profile text**")
                snippet = row["profile_text"]
                st.write(snippet[:1500] + ("..." if len(snippet) > 1500 else ""))
            with right:
                st.metric("Grad year", row["grad_year"])
                st.metric("Region", row["region"])
                st.metric("Availability", row["availability"])
