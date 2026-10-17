"""Sentence embeddings for every post (all-MiniLM-L6-v2, CPU-friendly).

Embeds the header plus the start of the body, which is where companies say what
they do. Cached incrementally: a monthly refresh only embeds new posts.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import EMBED_MODEL, INTERIM
from .parse import strip_urls

log = logging.getLogger(__name__)

CACHE = INTERIM / "embeddings.npz"
MAX_CHARS = 1200
CHUNK = 4000
# Bump when doc_text changes so stale cached vectors are discarded.
# v1 embedded the header line too; clusters then formed around "City | REMOTE | Full-time"
# formatting and geography instead of what the company does.
DOC_VERSION = 2


def doc_text(text: str) -> str:
    """The post body without its metadata lines (Company | Role | City | REMOTE ...)."""
    lines = strip_urls(text).split("\n")
    body = [ln for ln in lines[1:] if ln.count("|") < 2]
    return ("\n".join(body) or lines[0])[:MAX_CHARS]


def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL, device="cpu")


def encode(model, texts: list[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True).astype(np.float32)


def run() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(INTERIM / "posts.parquet", columns=["post_id", "text"])
    ids, vecs = np.array([], dtype=np.int64), np.zeros((0, 384), dtype=np.float32)
    if CACHE.exists():
        z = np.load(CACHE)
        cached_version = int(z["version"]) if "version" in z.files else 1
        if cached_version == DOC_VERSION:
            ids, vecs = z["post_id"], z["emb"]
        else:
            log.info("Embedding cache is from an older doc_text version; re-embedding")

    todo = df[~df["post_id"].isin(ids)]
    if len(todo):
        log.info("Embedding %d new posts (%d cached)", len(todo), len(ids))
        model = load_model()
        # Checkpoint every chunk so an interrupted run resumes instead of restarting.
        for start in range(0, len(todo), CHUNK):
            part = todo.iloc[start:start + CHUNK]
            new = encode(model, [doc_text(t) for t in part["text"]])
            ids = np.concatenate([ids, part["post_id"].to_numpy()])
            vecs = np.vstack([vecs, new])
            np.savez(CACHE, post_id=ids, emb=vecs, version=DOC_VERSION)
            log.info("  %d / %d", min(start + CHUNK, len(todo)), len(todo))

    order = pd.Series(np.arange(len(ids)), index=ids).reindex(df["post_id"]).to_numpy()
    return df["post_id"].to_numpy(), vecs[order]


def load() -> tuple[np.ndarray, np.ndarray]:
    z = np.load(CACHE)
    return z["post_id"], z["emb"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()
