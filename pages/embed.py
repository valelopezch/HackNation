# talentai/embed.py
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer(MODEL_NAME)

def embed_texts(texts: list[str]) -> np.ndarray:
    model = load_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)
