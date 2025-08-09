# talentai/skills.py
from __future__ import annotations
from typing import Iterable, List, Dict
import pandas as pd

def load_taxonomy(csv_path: str) -> pd.DataFrame:
    # Same format as parse.load_skill_lexicon
    return pd.read_csv(csv_path)

def normalize(skills_found: Iterable[str], taxonomy: pd.DataFrame) -> List[str]:
    canon = taxonomy["skill"].unique().tolist()
    # We assume parse already mapped to canon; keep intersection only
    return sorted(set(s for s in skills_found if s in canon))
