# pages/_candidate.py
import streamlit as st
from pages.parse import parse_cv
from pages.skills import load_taxonomy, normalize
from pages.grading import grade_debug_submission, grade_plot_matching, BUGGY_SNIPPET, PLOTS

def render():
    st.set_page_config(layout="wide")
    st.title("Candidate – Profile & Mini‑challenge")

    tax = load_taxonomy("data/skills.csv")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Upload CV (PDF)")
        up = st.file_uploader("PDF only", type=["pdf"])
        if up:
            pdf_path = f"/tmp/{up.name}"
            with open(pdf_path, "wb") as f: f.write(up.read())
            parsed = parse_cv(pdf_path, "data/skills.csv")
            st.session_state["cand"] = parsed
            st.success("Parsed!")
    with col2:
        if "cand" in st.session_state:
            cand = st.session_state["cand"]
            st.write("**Detected skills**:", ", ".join(cand["skills"]) or "—")
            add = st.text_input("Add skills (comma‑separated)")
            if add:
                added = [a.strip() for a in add.split(",") if a.strip()]
                cand["skills"] = sorted(set(cand["skills"]) | set(normalize(added, tax)))
            st.write("**Final skills**:", ", ".join(cand["skills"]) or "—")

    st.divider()
    st.subheader("Mini‑challenge A: Model Debug Sprint")
    st.caption("Fix the classification bug (no external data/APIs). Paste adjusted code; we check heuristics.")
    with st.expander("Show buggy starter"):
        st.code(BUGGY_SNIPPET, language="python")
    code = st.text_area("Paste your fixed snippet here", height=220, key="codeA")
    if st.button("Grade A"):
        res = grade_debug_submission(code)
        st.metric("Score", f"{res.score}/{res.max_score}")
        st.write(res.feedback or "Great job!")

    st.subheader("Mini‑challenge B: Match model to plot")
    answers = {}
    cols = st.columns(4)
    for i, (img, correct) in enumerate(PLOTS.items()):
        with cols[i]:
            st.image(f"data/{img}", caption=f"Plot {i+1}")
            answers[img] = st.selectbox(
                f"Model {i+1}",
                ["", "Logistic Regression", "kNN", "Decision Tree", "SVM (RBF)"],
                key=f"m{i}",
            )
    if st.button("Grade B"):
        res = grade_plot_matching(answers)
        st.metric("Score", f"{res.score}/{res.max_score}")
        st.write(res.feedback)
