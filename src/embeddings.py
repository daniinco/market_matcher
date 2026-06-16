"""Module: Compute and cache GigaChatEmbeddings for all supplier product names.

Uses GigaChatEmbeddings from ffd.py pattern.
Saves/loads embeddings from output/embeddings_cache.npz for reuse.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_gigachat.embeddings import GigaChatEmbeddings

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
load_dotenv(override=True)

CACHE_PATH = Path(__file__).parent.parent / "output" / "embeddings_cache.npz"
BATCH_SIZE = 50   # texts per embed_documents call


# ── Embedder initialization (pattern from ffd.py:43) ──────────────────────────

def get_embedder() -> GigaChatEmbeddings:
    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        raise RuntimeError("GIGACHAT_CREDENTIALS не найден. Проверьте .env.")
    return GigaChatEmbeddings(
        credentials=credentials.strip("'\""),
        base_url=os.getenv("GIGACHAT_BASE_URL", "").strip("'\"") or None,
        auth_url=os.getenv("GIGACHAT_AUTH_URL", "").strip("'\"") or None,
        scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS").strip("'\""),
        verify_ssl_certs=False,
        profanity_check=False,
    )


# ── Batch embedding ────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], embedder: GigaChatEmbeddings) -> np.ndarray:
    """Embed a list of texts in batches. Returns (N, D) float32 array."""
    all_vecs: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, BATCH_SIZE):
        batch = texts[start: start + BATCH_SIZE]
        vecs = embedder.embed_documents(batch)
        all_vecs.extend(vecs)
        print(f"  Embedded {min(start + BATCH_SIZE, total)}/{total} texts")
    return np.array(all_vecs, dtype=np.float32)


# ── Cache management ───────────────────────────────────────────────────────────

def _cache_key(filename: str) -> str:
    """Sanitize filename for use as npz key (no dots/slashes)."""
    return filename.replace(".", "_").replace("/", "_").replace("\\", "_")


def save_cache(embeddings: dict[str, np.ndarray], row_ids: dict[str, list[int]]) -> None:
    """Save embeddings and row_id mappings to NPZ cache."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, np.ndarray] = {}
    for fname, vecs in embeddings.items():
        key = _cache_key(fname)
        save_dict[f"emb_{key}"] = vecs
        save_dict[f"ids_{key}"] = np.array(row_ids[fname], dtype=np.int32)
    np.savez_compressed(CACHE_PATH, **save_dict)
    print(f"  Embeddings cache saved → {CACHE_PATH}")


def load_cache() -> tuple[dict[str, np.ndarray], dict[str, list[int]]] | None:
    """Load embeddings cache. Returns (embeddings, row_ids) or None if not found."""
    if not CACHE_PATH.exists():
        return None
    data = np.load(CACHE_PATH, allow_pickle=False)
    embeddings: dict[str, np.ndarray] = {}
    row_ids: dict[str, list[int]] = {}

    # Reconstruct original filenames from sanitized keys
    emb_keys = [k for k in data.files if k.startswith("emb_")]
    for emb_key in emb_keys:
        sanitized = emb_key[4:]  # strip "emb_"
        ids_key = f"ids_{sanitized}"
        # Reverse sanitization: find original filename
        # We store the sanitized key and match by pattern
        embeddings[sanitized] = data[emb_key]
        row_ids[sanitized] = data[ids_key].tolist()

    return embeddings, row_ids


def _sanitized_to_original(sanitized: str, filenames: list[str]) -> str | None:
    """Find original filename matching a sanitized cache key."""
    for fname in filenames:
        if _cache_key(fname) == sanitized:
            return fname
    return None


# ── Main public API ────────────────────────────────────────────────────────────

def compute_embeddings(
    supplier_data: dict[str, pd.DataFrame],
    force_recompute: bool = False,
) -> dict[str, tuple[np.ndarray, list[int]]]:
    """Compute or load cached embeddings for all supplier product names.

    Returns dict: filename -> (embedding_matrix, row_ids_list)
    embedding_matrix shape: (N, D) float32
    row_ids_list: list of row_id values matching rows of embedding_matrix
    """
    filenames = list(supplier_data.keys())

    # Try loading from cache
    if not force_recompute and CACHE_PATH.exists():
        cached = load_cache()
        if cached is not None:
            raw_embeddings, raw_ids = cached
            # Map sanitized keys back to original filenames
            result: dict[str, tuple[np.ndarray, list[int]]] = {}
            cache_ok = True
            for fname in filenames:
                skey = _cache_key(fname)
                if skey in raw_embeddings:
                    cached_n = raw_embeddings[skey].shape[0]
                    expected_n = len(supplier_data[fname])
                    if cached_n < expected_n:
                        print(
                            f"  Cache mismatch for {fname}: "
                            f"cache has {cached_n} vectors but data has {expected_n} rows — recomputing"
                        )
                        cache_ok = False
                        break
                    result[fname] = (raw_embeddings[skey], raw_ids[skey])
                else:
                    print(f"  WARNING: {fname} not in cache, will recompute")
                    cache_ok = False
                    break
            if cache_ok and len(result) == len(filenames):
                total_vecs = sum(v[0].shape[0] for v in result.values())
                print(f"  Loaded embeddings from cache: {total_vecs} vectors")
                return result

    # Compute embeddings
    print("  Initializing GigaChatEmbeddings...")
    embedder = get_embedder()

    result = {}
    all_embeddings: dict[str, np.ndarray] = {}
    all_row_ids: dict[str, list[int]] = {}

    for fname, df in supplier_data.items():
        print(f"\n  Computing embeddings for {fname} ({len(df)} rows)...")
        texts = df["product_name"].tolist()
        ids = df["row_id"].tolist()
        vecs = embed_texts(texts, embedder)
        result[fname] = (vecs, ids)
        all_embeddings[fname] = vecs
        all_row_ids[fname] = ids

    save_cache(all_embeddings, all_row_ids)
    return result


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between all pairs of rows in a and b.

    Args:
        a: (M, D) float32
        b: (N, D) float32

    Returns:
        (M, N) float32 cosine similarity matrix
    """
    # L2-normalize
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from load_data import load_all_suppliers

    print("Loading data (first 10 rows for test)...")
    data = load_all_suppliers(max_rows=10)

    print("\nComputing embeddings...")
    emb_data = compute_embeddings(data, force_recompute=True)

    for fname, (vecs, ids) in emb_data.items():
        print(f"  {fname}: shape={vecs.shape}  row_ids={ids[:3]}...")

    # Test cosine similarity between first two files
    fnames = list(emb_data.keys())
    vecs_a, ids_a = emb_data[fnames[0]]
    vecs_b, ids_b = emb_data[fnames[1]]
    sim = cosine_similarity_matrix(vecs_a, vecs_b)
    print(f"\nCosine similarity matrix {fnames[0].split('_')[1]} × {fnames[1].split('_')[1]}: shape={sim.shape}")
    print(f"Max similarity: {sim.max():.4f}  Mean: {sim.mean():.4f}")
    i, j = np.unravel_index(sim.argmax(), sim.shape)
    print(f"Most similar pair: row_id {ids_a[i]} × {ids_b[j]}  sim={sim[i,j]:.4f}")
