# talentai/parse.py FEFI
from __future__ import annotations
import re, json
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Set
from rapidfuzz import process, fuzz
import pandas as pd

SKILL_COL = "skill"

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    doc = fitz.open(str(pdf_path))
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text

def load_skill_lexicon(csv_path: str | Path) -> pd.DataFrame:
    # CSV columns: skill,synonyms (pipe-separated)
    df = pd.read_csv(csv_path)
    df["synonyms"] = df["synonyms"].fillna("").astype(str)
    return df

def extract_skills_freeform(text: str, lex: pd.DataFrame, top_k_per_syn=3) -> List[str]:
    # Build search list = skill + synonyms
    candidates = []
    for _, row in lex.iterrows():
        names = [row[SKILL_COL]] + [s.strip() for s in row["synonyms"].split("|") if s.strip()]
        for name in names:
            candidates.append(name)
    # Simple case-insensitive keyword scan + fuzzy bump
    found = set()
    lowtext = text.lower()
    for cand in set(candidates):
        if len(cand) < 2: 
            continue
        if cand.lower() in lowtext:
            found.add(cand)
    # Map found → canonical via best match
    canonical = []
    canon_list = lex[SKILL_COL].unique().tolist()
    for f in found:
        best = process.extractOne(f, canon_list, scorer=fuzz.WRatio)
        if best and best[1] >= 85:
            canonical.append(best[0])
    return sorted(set(canonical))

def parse_cv(pdf_path: str | Path, skills_csv: str | Path) -> Dict:
    text = extract_text_from_pdf(pdf_path)
    lex = load_skill_lexicon(skills_csv)
    skills = extract_skills_freeform(text, lex)
    # naive metadata pulls
    location = re.search(r"(Based in|Location)[:\- ]+([A-Za-z ,]+)", text, re.I)
    grad = re.search(r"(Graduation|Graduated)[^0-9]*(20\d{2})", text, re.I)
    return {
        "text": text,
        "skills": skills,
        "location": location.group(2).strip() if location else "",
        "grad_year": int(grad.group(2)) if grad else None
    }
