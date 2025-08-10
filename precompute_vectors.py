# precompute_vectors.py
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# --- usa tus utilidades existentes ---
from data_utils import (
    load_jobs, load_candidates, candidate_skills_map
)

# IMPORTANTE: estas helpers/corpus vienen de embed.py (según lo que ya integramos)
from embed import (
    build_job_corpus, build_candidate_text, build_candidate_corpus,
    df_sha256, config_sha256, store_dir_for, write_manifest
)

# GPU (auto) para sentence-transformers
try:
    import torch
    from sentence_transformers import SentenceTransformer
    TORCH_OK = True
except Exception:
    TORCH_OK = False


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def fit_tfidf_and_save(corpus: pd.Series, out_dir: Path, mat_filename: str):
    vec_path = out_dir / "tfidf_vectorizer.joblib"
    mat_path = out_dir / mat_filename

    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vectorizer.fit_transform(corpus).toarray().astype(np.float32)

    joblib.dump(vectorizer, vec_path)
    np.save(mat_path, X)
    return vectorizer, X


def encode_and_build_faiss(corpus: pd.Series, model_name: str, out_path: Path,
                           device: str = "auto", batch_size: int = 128,
                           normalize: bool = True, show_progress: bool = True):
    if not TORCH_OK:
        raise RuntimeError("PyTorch / sentence-transformers no disponibles en este entorno.")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)

    # Barra de progreso integrada de sentence-transformers
    emb = model.encode(
        corpus.tolist(),
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=normalize
    ).astype(np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(out_path))
    return index, emb


def main():
    ap = argparse.ArgumentParser(description="Precompute TF-IDF and FAISS indexes for jobs & candidates.")
    ap.add_argument("--jobs_csv", default=None, help="Ruta a jobs CSV (si no se pasa, usa load_jobs())")
    ap.add_argument("--cands_csv", default=None, help="Ruta a candidates CSV (si no se pasa, usa load_candidates())")
    ap.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2", help="Modelo de sentence-transformers")
    ap.add_argument("--store_dir", default="./vector_store", help="Directorio raíz del vector store")
    ap.add_argument("--batch_size", type=int, default=128, help="Batch size para embeddings")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Dispositivo para embeddings")
    ap.add_argument("--no_normalize", action="store_true", help="No normalizar embeddings (por defecto normaliza)")
    args = ap.parse_args()

    normalize = not args.no_normalize

    # 1) Carga datos
    if args.jobs_csv:
        jobs = pd.read_csv(args.jobs_csv).fillna("")
    else:
        jobs = load_jobs()

    if args.cands_csv:
        cands = pd.read_csv(args.cands_csv).fillna("")
    else:
        cands = load_candidates()

    # skills_map para el corpus de candidatos
    skills_map = candidate_skills_map(cands)

    # 2) Config y firmas (deben MATCHEAR embed.py / DualMatcher)
    cfg = {"model_name": args.model, "tfidf_min_df": 1, "tfidf_ngram": (1, 2)}
    cfg_sig = config_sha256(cfg)
    jobs_sig = df_sha256(jobs)
    cands_sig = df_sha256(cands)

    # 3) Deriva carpetas de store
    store_root = Path(args.store_dir)
    jobs_dir = store_dir_for(store_root, "jobs", jobs_sig, cfg_sig)
    cands_dir = store_dir_for(store_root, "cands", cands_sig, cfg_sig)

    print(f"[jobs] store: {jobs_dir}")
    print(f"[cands] store: {cands_dir}")

    # 4) Construye corpus
    jobs_corpus = build_job_corpus(jobs)
    cands_corpus = build_candidate_corpus(cands, skills_map)

    # 5) TF-IDF + matrices
    print("\n[1/3] Fitting TF-IDF (jobs)…")
    jvec, jX = fit_tfidf_and_save(jobs_corpus, jobs_dir, "jobs_tfidf.npy")
    print("     ->", (jX.shape[0], jX.shape[1]))

    print("[2/3] Fitting TF-IDF (cands)…")
    ccvec, cX = fit_tfidf_and_save(cands_corpus, cands_dir, "cands_tfidf.npy")
    print("     ->", (cX.shape[0], cX.shape[1]))

    # 6) Embeddings + FAISS (con barra de progreso y GPU si hay)
    print("\n[3/3] Encoding embeddings + building FAISS (jobs)…")
    jfaiss_path = jobs_dir / "jobs_faiss.index"
    jfaiss, jemb = encode_and_build_faiss(
        jobs_corpus, args.model, jfaiss_path,
        device=args.device, batch_size=args.batch_size,
        normalize=normalize, show_progress=True
    )
    print(f"     -> FAISS ntotal={jfaiss.ntotal}, dim={jemb.shape[1]}")

    print("Encoding embeddings + building FAISS (cands)…")
    cfaiss_path = cands_dir / "cands_faiss.index"
    cfaiss, cemb = encode_and_build_faiss(
        cands_corpus, args.model, cfaiss_path,
        device=args.device, batch_size=args.batch_size,
        normalize=normalize, show_progress=True
    )
    print(f"     -> FAISS ntotal={cfaiss.ntotal}, dim={cemb.shape[1]}")

    # 7) Manifest
    from embed import write_manifest  # ya importado arriba; solo recordatorio
    write_manifest(jobs_dir, {
        "sha256": jobs_sig,
        "rows": int(len(jobs)),
        "model_name": args.model,
        "tfidf_min_df": 1,
        "tfidf_ngram": (1, 2),
        "faiss_ntotal": int(jfaiss.ntotal)
    })
    write_manifest(cands_dir, {
        "sha256": cands_sig,
        "rows": int(len(cands)),
        "model_name": args.model,
        "tfidf_min_df": 1,
        "tfidf_ngram": (1, 2),
        "faiss_ntotal": int(cfaiss.ntotal)
    })

    print("\n✅ Precompute listo. Puedes lanzar Streamlit; DualMatcher usará estos índices.")


if __name__ == "__main__":
    main()
