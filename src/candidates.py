"""Module 2: Generate candidate pairs using vectorized price-window + embedding cosine blocking.

Two modes:
  - With embeddings (preferred): price window + cosine similarity on GigaChatEmbeddings
  - Without embeddings (fallback): price window + token Jaccard

Uses only pandas + numpy (no sklearn dependency).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

# ── Blocking thresholds ────────────────────────────────────────────────────────
PRICE_WINDOW      = 0.30   # |price_A - price_B| / price_A <= 30%

# Embedding-based thresholds
COSINE_MIN        = 0.89   # cosine similarity >= 0.89
COSINE_TIGHT      = 0.85   # OR cosine >= 0.85 when price diff <= 2%
PRICE_TIGHT_EMB   = 0.02   # price diff <= 2% for tight cosine path

# Jaccard fallback thresholds
JACCARD_MIN       = 0.20   # token Jaccard >= 0.20
JACCARD_TIGHT     = 0.05   # OR Jaccard >= 0.05 when price diff <= 2%
PRICE_TIGHT_JAC   = 0.02   # price diff <= 2% for tight Jaccard path

MAX_CANDIDATES    = 130    # hard cap per file-pair


@dataclass
class CandidatePair:
    left_file: str
    left_row_id: int
    left_name: str
    left_price: float
    right_file: str
    right_row_id: int
    right_name: str
    right_price: float
    price_diff_pct: float
    token_jaccard: float        # cosine sim when embeddings used, Jaccard otherwise
    intra_supplier: bool = False


# ── Token helpers (Jaccard fallback) ──────────────────────────────────────────

def _tokenize(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-zа-яё0-9]+", text.lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ── Core blocking ──────────────────────────────────────────────────────────────

def _cross_join_price_filter(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    intra: bool,
) -> pd.DataFrame:
    """Stage 1: cross-join + price window filter. Returns merged DataFrame."""
    a = df_a[["row_id", "product_name", "product_name_norm", "price_rub", "file"]].copy()
    b = df_b[["row_id", "product_name", "product_name_norm", "price_rub", "file"]].copy()
    a.columns = ["left_row_id", "left_name", "left_norm", "left_price", "left_file"]
    b.columns = ["right_row_id", "right_name", "right_norm", "right_price", "right_file"]

    a = a.assign(_k=1)
    b = b.assign(_k=1)
    merged = a.merge(b, on="_k").drop(columns="_k")

    if intra:
        merged = merged[merged["left_row_id"] < merged["right_row_id"]]

    if merged.empty:
        return merged

    left_p = merged["left_price"].to_numpy(dtype=float)
    right_p = merged["right_price"].to_numpy(dtype=float)
    denom = np.where(left_p > 0, left_p, 1.0)
    pdiff = np.abs(left_p - right_p) / denom
    merged = merged[pdiff <= PRICE_WINDOW].copy()
    merged["price_diff"] = pdiff[pdiff <= PRICE_WINDOW]

    return merged


def _generate_pairs_with_embeddings(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    emb_a: np.ndarray,
    ids_a: list[int],
    emb_b: np.ndarray,
    ids_b: list[int],
    intra: bool,
) -> list[CandidatePair]:
    """Generate candidates using embedding cosine similarity after price filter."""
    from embeddings import cosine_similarity_matrix

    merged = _cross_join_price_filter(df_a, df_b, intra)
    if merged.empty:
        return []

    # Build row_id → embedding index maps
    id_to_idx_a = {rid: i for i, rid in enumerate(ids_a)}
    id_to_idx_b = {rid: i for i, rid in enumerate(ids_b)}

    # Get embedding indices for each row in merged
    left_indices  = merged["left_row_id"].map(id_to_idx_a).to_numpy()
    right_indices = merged["right_row_id"].map(id_to_idx_b).to_numpy()

    # Filter out rows where embedding index is missing
    valid_mask = (~np.isnan(left_indices.astype(float))) & (~np.isnan(right_indices.astype(float)))
    if not valid_mask.all():
        merged = merged[valid_mask].copy()
        left_indices  = left_indices[valid_mask]
        right_indices = right_indices[valid_mask]

    if merged.empty:
        return []

    left_indices  = left_indices.astype(int)
    right_indices = right_indices.astype(int)

    # Compute per-row cosine similarity (no full matrix needed)
    left_vecs  = emb_a[left_indices]   # (M, D)
    right_vecs = emb_b[right_indices]  # (M, D)

    # Row-wise cosine: dot(normalize(a), normalize(b))
    left_norm  = left_vecs  / (np.linalg.norm(left_vecs,  axis=1, keepdims=True) + 1e-9)
    right_norm = right_vecs / (np.linalg.norm(right_vecs, axis=1, keepdims=True) + 1e-9)
    cosines = (left_norm * right_norm).sum(axis=1)

    merged["cosine"] = cosines
    pdiff_arr = merged["price_diff"].to_numpy()

    # Filter: cosine >= COSINE_MIN OR (price_diff <= PRICE_TIGHT_EMB AND cosine >= COSINE_TIGHT)
    mask = (cosines >= COSINE_MIN) | (
        (pdiff_arr <= PRICE_TIGHT_EMB) & (cosines >= COSINE_TIGHT)
    )
    merged = merged[mask].copy()

    if merged.empty:
        return []

    merged = merged.sort_values("cosine", ascending=False).head(MAX_CANDIDATES)

    pairs: list[CandidatePair] = []
    for row in merged.itertuples(index=False):
        pairs.append(CandidatePair(
            left_file=row.left_file,
            left_row_id=int(row.left_row_id),
            left_name=row.left_name,
            left_price=float(row.left_price),
            right_file=row.right_file,
            right_row_id=int(row.right_row_id),
            right_name=row.right_name,
            right_price=float(row.right_price),
            price_diff_pct=round(float(row.price_diff) * 100, 2),
            token_jaccard=round(float(row.cosine), 4),
            intra_supplier=intra,
        ))
    return pairs


def _generate_pairs_jaccard(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    intra: bool,
) -> list[CandidatePair]:
    """Generate candidates using token Jaccard similarity (fallback, no embeddings)."""
    merged = _cross_join_price_filter(df_a, df_b, intra)
    if merged.empty:
        return []

    all_norms = pd.concat([merged["left_norm"], merged["right_norm"]]).unique()
    token_cache: dict[str, frozenset] = {n: _tokenize(n) for n in all_norms}

    left_tokens  = merged["left_norm"].map(token_cache).to_numpy()
    right_tokens = merged["right_norm"].map(token_cache).to_numpy()

    jaccards = np.array([
        _jaccard(lt, rt)
        for lt, rt in zip(left_tokens, right_tokens)
    ])

    merged["jaccard"] = jaccards
    pdiff_arr = merged["price_diff"].to_numpy()

    mask = (jaccards >= JACCARD_MIN) | (
        (pdiff_arr <= PRICE_TIGHT_JAC) & (jaccards >= JACCARD_TIGHT)
    )
    merged = merged[mask].copy()

    if merged.empty:
        return []

    merged = merged.sort_values("jaccard", ascending=False).head(MAX_CANDIDATES)

    pairs: list[CandidatePair] = []
    for row in merged.itertuples(index=False):
        pairs.append(CandidatePair(
            left_file=row.left_file,
            left_row_id=int(row.left_row_id),
            left_name=row.left_name,
            left_price=float(row.left_price),
            right_file=row.right_file,
            right_row_id=int(row.right_row_id),
            right_name=row.right_name,
            right_price=float(row.right_price),
            price_diff_pct=round(float(row.price_diff) * 100, 2),
            token_jaccard=round(float(row.jaccard), 4),
            intra_supplier=intra,
        ))
    return pairs


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_candidates(
    supplier_data: dict[str, pd.DataFrame],
    include_intra: bool = True,
    embedding_data: dict[str, tuple[np.ndarray, list[int]]] | None = None,
) -> list[CandidatePair]:
    """Generate all candidate pairs across all supplier file combinations.

    Args:
        supplier_data: dict returned by load_all_suppliers()
        include_intra: if True, also detect duplicates within the same supplier file
        embedding_data: optional dict from compute_embeddings(); if provided, uses
                        cosine similarity instead of Jaccard for Stage 2 filtering

    Returns:
        list of CandidatePair
    """
    use_embeddings = embedding_data is not None
    mode_label = "cosine(embeddings)" if use_embeddings else "Jaccard(tokens)"
    print(f"  Blocking mode: price-window ±30% + {mode_label}")

    all_pairs: list[CandidatePair] = []
    filenames = list(supplier_data.keys())

    for fname_a, fname_b in combinations(filenames, 2):
        df_a = supplier_data[fname_a]
        df_b = supplier_data[fname_b]

        if use_embeddings and fname_a in embedding_data and fname_b in embedding_data:
            emb_a, ids_a = embedding_data[fname_a]
            emb_b, ids_b = embedding_data[fname_b]
            pairs = _generate_pairs_with_embeddings(
                df_a, df_b, emb_a, ids_a, emb_b, ids_b, intra=False
            )
        else:
            pairs = _generate_pairs_jaccard(df_a, df_b, intra=False)

        label_a = fname_a.split("_")[1] if "_" in fname_a else fname_a
        label_b = fname_b.split("_")[1] if "_" in fname_b else fname_b
        print(f"  {label_a} × {label_b}: {len(pairs)} candidates")
        all_pairs.extend(pairs)

    if include_intra:
        for fname, df in supplier_data.items():
            if use_embeddings and fname in embedding_data:
                emb, ids = embedding_data[fname]
                pairs = _generate_pairs_with_embeddings(
                    df, df, emb, ids, emb, ids, intra=True
                )
            else:
                pairs = _generate_pairs_jaccard(df, df, intra=True)
            label = fname.split("_")[1] if "_" in fname else fname
            print(f"  {label} (intra): {len(pairs)} candidates")
            all_pairs.extend(pairs)

    print(f"\nTotal candidates: {len(all_pairs)}")
    return all_pairs


def get_candidates_for_pair(
    supplier_data: dict[str, pd.DataFrame],
    left_file: str,
    left_row_id: int,
    right_file: str,
    right_row_ids: list[int] | None = None,
    embedding_data: dict[str, tuple[np.ndarray, list[int]]] | None = None,
) -> list[CandidatePair]:
    """Get candidates for a specific query row (used by QA evaluation).

    Falls back to price-window-only if no similarity matches found.
    """
    df_left = supplier_data[left_file]
    df_right = supplier_data[right_file]

    row_left = df_left[df_left["row_id"] == left_row_id].copy()
    if row_left.empty:
        return []

    if right_row_ids is not None:
        df_right = df_right[df_right["row_id"].isin(right_row_ids)].copy()

    intra = left_file == right_file

    if (
        embedding_data is not None
        and left_file in embedding_data
        and right_file in embedding_data
    ):
        emb_l, ids_l = embedding_data[left_file]
        emb_r, ids_r = embedding_data[right_file]
        pairs = _generate_pairs_with_embeddings(
            row_left, df_right, emb_l, ids_l, emb_r, ids_r, intra=intra
        )
    else:
        pairs = _generate_pairs_jaccard(row_left, df_right, intra=intra)

    if not pairs:
        # Fallback: price-window only, no similarity filter
        merged = _cross_join_price_filter(row_left, df_right, intra=intra)
        for row in merged.itertuples(index=False):
            ta = _tokenize(row.left_norm)
            tb = _tokenize(row.right_norm)
            pairs.append(CandidatePair(
                left_file=row.left_file,
                left_row_id=int(row.left_row_id),
                left_name=row.left_name,
                left_price=float(row.left_price),
                right_file=row.right_file,
                right_row_id=int(row.right_row_id),
                right_name=row.right_name,
                right_price=float(row.right_price),
                price_diff_pct=round(float(row.price_diff) * 100, 2),
                token_jaccard=round(_jaccard(ta, tb), 4),
                intra_supplier=intra,
            ))
    return pairs


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from load_data import load_all_suppliers

    print("Loading data...")
    data = load_all_suppliers(max_rows=50)
    print("Generating candidates (Jaccard fallback)...")
    t0 = time.time()
    candidates = generate_candidates(data, include_intra=False)
    print(f"Done in {time.time()-t0:.1f}s  Total: {len(candidates)}")
    for c in candidates[:3]:
        print(f"  {c.left_file.split('_')[1]}:{c.left_row_id} | {c.left_name[:40]!r}")
        print(f"  {c.right_file.split('_')[1]}:{c.right_row_id} | {c.right_name[:40]!r}")
        print(f"  price_diff={c.price_diff_pct}%  sim={c.token_jaccard}")
        print()
