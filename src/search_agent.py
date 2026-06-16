"""LangGraph Search Agent — find products across supplier catalogues by free-text query.

Flow:
  clarification_node  → detect vague queries, ask one follow-up question
  normalize_node      → rewrite conversational query to product description
  search_node         → embed query, cosine top-15 per supplier file
  judge_node          → GigaChat judges each hit: does it match the query?
  check_count_node    → route: ≥8 found → format; else → expand outliers
  expand_node         → fetch ranks 16-25 from 3 most-distant non-match files
  judge_expanded_node → judge expanded hits
  format_node         → expand via clusters, deduplicate, print results
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
load_dotenv(override=True)

# Ensure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from embeddings import compute_embeddings, cosine_similarity_matrix, get_embedder
from judge import get_llm
from load_data import load_all_suppliers

OUTPUT_DIR = Path(__file__).parent.parent / "output"
CLUSTERS_CSV = OUTPUT_DIR / "clusters_predicted.csv"

TOP_K = 15          # candidates per file in initial search
EXPAND_K = 10       # extra candidates per file in expansion
MIN_RESULTS = 8     # stop early if this many matches found

# System prompt for the judge role
JUDGE_SYSTEM_PROMPT = (
    "Ты эксперт по подбору мебели и товаров для дома. "
    "Твоя задача — определить, соответствует ли товар из каталога запросу пользователя. "
    "Учитывай тип товара, размеры, материал, цвет и стиль. "
    "Отвечай строго JSON без markdown."
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SearchHit:
    file: str
    row_id: int
    product_name: str
    price_rub: float
    cosine_score: float


@dataclass
class ResultItem:
    file: str
    row_id: int
    product_name: str
    price_rub: float
    source: str          # "direct" | "cluster"
    cluster_id: str = ""


# ── LangGraph State ────────────────────────────────────────────────────────────

class SearchAgentState(TypedDict):
    raw_query: str
    clarification_answer: str    # user's answer to clarifying question (empty if not needed)
    normalized_query: str
    supplier_data: dict          # dict[str, pd.DataFrame]
    embedding_data: dict         # dict[str, (np.ndarray, list[int])]
    top15_per_file: dict         # dict[str, list[SearchHit]]
    accepted: list               # list[SearchHit]
    rejected: list               # list[SearchHit]
    outliers: list               # list[SearchHit]  — 3 most-distant non-matches
    expanded_hits: list          # list[SearchHit]
    final_results: list          # list[ResultItem]
    phase: str                   # "initial" | "expanded"


# ── Node 0: clarification_node ────────────────────────────────────────────────

def clarification_node(state: SearchAgentState) -> SearchAgentState:
    """Detect vague queries and ask one clarifying question before normalization."""
    raw = state["raw_query"]
    print(f"\n[0/6] Checking if query needs clarification: {raw!r}")

    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "Ты помощник по поиску мебели и товаров для дома. "
            "Определи, является ли запрос пользователя слишком общим — "
            "например, просто 'стол', 'кровать', 'шкаф', 'диван', 'кресло' без каких-либо уточнений "
            "(размера, цвета, материала, стиля, назначения). "
            "Если запрос слишком общий — сформулируй ОДИН короткий уточняющий вопрос. "
            "Если запрос уже достаточно конкретный — верни пустую строку в поле question. "
            "Ответь строго JSON без markdown: "
            '{\"is_vague\": true/false, \"question\": \"...\"}'
        )),
        HumanMessage(content=raw),
    ]

    is_vague = False
    question = ""
    try:
        response = llm.invoke(messages)
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        is_vague = bool(data.get("is_vague", False))
        question = str(data.get("question", "")).strip()
    except Exception as e:
        print(f"  WARNING: clarification check failed ({e}), skipping")

    clarification_answer = ""
    enriched_query = raw

    if is_vague and question:
        print(f"\n  Запрос слишком общий. Уточняющий вопрос:")
        print(f"  {question}")
        try:
            answer = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer:
            clarification_answer = answer
            enriched_query = f"{raw}, {answer}"
            print(f"  Уточнённый запрос: {enriched_query!r}")
        else:
            print("  Ответ не получен, продолжаем с исходным запросом")
    else:
        print("  Запрос достаточно конкретный, уточнение не требуется")

    return {
        **state,
        "raw_query": enriched_query,
        "clarification_answer": clarification_answer,
    }


# ── Node 1: normalize_node ─────────────────────────────────────────────────────

def normalize_node(state: SearchAgentState) -> SearchAgentState:
    """Rewrite conversational query to a clean product description."""
    raw = state["raw_query"]
    print(f"\n[1/6] Normalizing query: {raw!r}")

    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "Ты помощник по поиску товаров в каталоге мебели и товаров для дома. "
            "Если запрос пользователя уже является описанием товара (название модели, "
            "характеристики, размеры, цвет) — верни его без изменений. "
            "Если это разговорный запрос — перефразируй в краткое описание товара. "
            "Верни ТОЛЬКО описание товара, без пояснений и кавычек."
        )),
        HumanMessage(content=raw),
    ]

    try:
        response = llm.invoke(messages)
        normalized = response.content.strip().strip('"\'')
    except Exception as e:
        print(f"  WARNING: normalization failed ({e}), using raw query")
        normalized = raw

    if normalized != raw:
        print(f"  Rewritten → {normalized!r}")
    else:
        print(f"  No rewrite needed")

    return {**state, "normalized_query": normalized}


# ── Node 2: search_node ────────────────────────────────────────────────────────

def search_node(state: SearchAgentState) -> SearchAgentState:
    """Embed the normalized query and find top-15 nearest neighbours per file."""
    query = state["normalized_query"]
    print(f"\n[2/6] Embedding query and searching {TOP_K} candidates per file...")

    embedder = get_embedder()
    query_vec = np.array(embedder.embed_query(query), dtype=np.float32).reshape(1, -1)

    supplier_data = state["supplier_data"]
    embedding_data = state["embedding_data"]

    top15_per_file: dict[str, list[SearchHit]] = {}

    for fname, df in supplier_data.items():
        if fname not in embedding_data:
            print(f"  WARNING: no embeddings for {fname}, skipping")
            continue

        vecs, row_ids = embedding_data[fname]
        vecs_arr = np.array(vecs, dtype=np.float32)

        # cosine similarity: (1, D) × (N, D)^T → (1, N)
        sim = cosine_similarity_matrix(query_vec, vecs_arr)[0]  # shape (N,)

        # top-K indices sorted descending
        top_idx = np.argsort(sim)[::-1][:TOP_K]

        hits: list[SearchHit] = []
        for idx in top_idx:
            rid = row_ids[idx]
            row = df[df["row_id"] == rid]
            if row.empty:
                continue
            row = row.iloc[0]
            hits.append(SearchHit(
                file=fname,
                row_id=int(rid),
                product_name=str(row["product_name"]),
                price_rub=float(row["price_rub"]) if pd.notna(row["price_rub"]) else 0.0,
                cosine_score=float(sim[idx]),
            ))

        top15_per_file[fname] = hits
        supplier_tag = fname.split("_")[1] if "_" in fname else fname
        print(f"  {supplier_tag}: top cosine={hits[0].cosine_score:.3f} … {hits[-1].cosine_score:.3f}")

    return {**state, "top15_per_file": top15_per_file}


# ── Shared judge helper ────────────────────────────────────────────────────────

def _judge_hits(
    hits: list[SearchHit],
    query: str,
    llm: Any,
    stop_at: int | None = None,
) -> tuple[list[SearchHit], list[SearchHit]]:
    """Judge a list of SearchHit objects. Returns (accepted, rejected).

    Args:
        stop_at: Stop judging as soon as this many hits are accepted (saves tokens).
                 Remaining hits are left unjudged.
    """
    accepted: list[SearchHit] = []
    rejected: list[SearchHit] = []

    for i, hit in enumerate(hits):
        # Early-exit: enough matches found — skip remaining hits
        if stop_at is not None and len(accepted) >= stop_at:
            remaining = len(hits) - i
            print(f"  Early stop: {len(accepted)} matches found, skipping {remaining} remaining hits")
            break

        prompt = (
            f"Пользователь ищет: {query}\n"
            f"Товар из каталога: {hit.product_name} — {hit.price_rub:.0f} руб.\n\n"
            "Этот товар подходит под запрос пользователя? "
            "Ответь строго JSON без markdown: "
            '{"match": true/false, "reason": "краткое объяснение"}'
        )
        try:
            response = llm.invoke([
                SystemMessage(content=JUDGE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
            text = response.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            is_match = bool(data.get("match", False))
        except Exception:
            # On parse failure, default to non-match
            is_match = False

        if is_match:
            accepted.append(hit)
        else:
            rejected.append(hit)

    return accepted, rejected


# ── Node 3: judge_node ─────────────────────────────────────────────────────────

def judge_node(state: SearchAgentState) -> SearchAgentState:
    """Judge top-15 hits per file with GigaChat, stopping early once MIN_RESULTS found."""
    query = state["normalized_query"]
    top15_per_file = state["top15_per_file"]

    all_hits: list[SearchHit] = []
    for hits in top15_per_file.values():
        all_hits.extend(hits)

    total = len(all_hits)
    print(f"\n[3/6] Judging up to {total} candidates with GigaChat (stop at {MIN_RESULTS} matches)...")

    llm = get_llm()
    accepted, rejected = _judge_hits(all_hits, query, llm, stop_at=MIN_RESULTS)

    print(f"  Accepted: {len(accepted)}  Rejected: {len(rejected)}")

    return {
        **state,
        "accepted": accepted,
        "rejected": rejected,
        "outliers": [],   # no longer computed here — expand_node uses top15_per_file directly
        "phase": "initial",
    }


# ── Node 4: check_count_node (routing) ────────────────────────────────────────

def check_count_node(state: SearchAgentState) -> SearchAgentState:
    """Decide whether to expand or format. Routing is done via conditional edge."""
    n = len(state["accepted"])
    print(f"\n[4/6] Check: {n} matches found (threshold={MIN_RESULTS})")
    if n >= MIN_RESULTS:
        print(f"  → Enough results, proceeding to format")
    elif state["phase"] == "initial":
        print(f"  → Too few results, will run per-file gate check + expansion")
    else:
        print(f"  → No more expansion possible, proceeding to format")
    return state


def route_after_check(state: SearchAgentState) -> str:
    """Conditional edge: expand or format."""
    if len(state["accepted"]) >= MIN_RESULTS:
        return "format"
    if state["phase"] == "initial":
        return "expand"
    return "format"


# ── Node 5: expand_node ────────────────────────────────────────────────────────

# Number of bottom hits per file to show GigaChat in the gate check
N_GATE_HITS = 3


def _gate_check_file(fname: str, bottom_hits: list[SearchHit], query: str, llm: Any) -> bool:
    """Ask GigaChat whether any of the bottom-ranked hits from a file are even
    remotely relevant to the query.  Returns True if the file is worth expanding.
    """
    items_text = "\n".join(
        f"  {i+1}. {h.product_name} — {h.price_rub:.0f} руб."
        for i, h in enumerate(bottom_hits)
    )
    prompt = (
        f"Пользователь ищет: {query}\n\n"
        f"Вот {len(bottom_hits)} товара из каталога поставщика:\n{items_text}\n\n"
        "Хотя бы один из этих товаров хоть отдалённо похож на то, что ищет пользователь? "
        "Ответь строго JSON без markdown: "
        '{"worth_expanding": true/false, "reason": "краткое объяснение"}'
    )
    try:
        response = llm.invoke([
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        text = response.content.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        return bool(data.get("worth_expanding", False))
    except Exception as e:
        # On failure, be conservative and expand anyway
        print(f"    WARNING: gate check failed ({e}), expanding by default")
        return True


def expand_node(state: SearchAgentState) -> SearchAgentState:
    """Per-file gate check: ask GigaChat if the 3 least-similar top-15 hits from
    each file are even remotely relevant.  Only fetch ranks 16-25 for files that
    pass the gate.
    """
    query_text = state["normalized_query"]
    top15_per_file = state["top15_per_file"]
    supplier_data = state["supplier_data"]
    embedding_data = state["embedding_data"]

    llm = get_llm()

    # Embed query (reuse embedder)
    embedder = get_embedder()
    query_vec = np.array(embedder.embed_query(query_text), dtype=np.float32).reshape(1, -1)

    # Track already-seen (file, row_id) pairs to avoid duplicates
    seen_pairs: set[tuple[str, int]] = set()
    for hits in top15_per_file.values():
        for h in hits:
            seen_pairs.add((h.file, h.row_id))

    print(f"\n[5/6] Per-file gate check + expansion (ranks {TOP_K+1}–{TOP_K+EXPAND_K})...")

    expanded_hits: list[SearchHit] = []
    files_expanded = 0
    files_skipped = 0

    for fname, hits in top15_per_file.items():
        supplier_tag = fname.split("_")[1] if "_" in fname else fname

        # Take the N_GATE_HITS least-similar hits from this file's top-15
        # (they are already sorted descending by cosine, so take the tail)
        bottom_hits = hits[-N_GATE_HITS:] if len(hits) >= N_GATE_HITS else hits

        # Gate check: is this file worth expanding?
        worth_it = _gate_check_file(fname, bottom_hits, query_text, llm)

        if not worth_it:
            print(f"  {supplier_tag}: gate=NO  → skipping")
            files_skipped += 1
            continue

        print(f"  {supplier_tag}: gate=YES → fetching ranks {TOP_K+1}–{TOP_K+EXPAND_K}")

        if fname not in embedding_data:
            continue

        df = supplier_data[fname]
        vecs, row_ids = embedding_data[fname]
        vecs_arr = np.array(vecs, dtype=np.float32)

        sim = cosine_similarity_matrix(query_vec, vecs_arr)[0]
        all_idx = np.argsort(sim)[::-1]

        count = 0
        for idx in all_idx:
            rid = int(row_ids[idx])
            if (fname, rid) in seen_pairs:
                continue
            row = df[df["row_id"] == rid]
            if row.empty:
                continue
            row = row.iloc[0]
            expanded_hits.append(SearchHit(
                file=fname,
                row_id=rid,
                product_name=str(row["product_name"]),
                price_rub=float(row["price_rub"]) if pd.notna(row["price_rub"]) else 0.0,
                cosine_score=float(sim[idx]),
            ))
            seen_pairs.add((fname, rid))
            count += 1
            if count >= EXPAND_K:
                break

        print(f"    → +{count} candidates added")
        files_expanded += 1

    print(f"  Summary: {files_expanded} files expanded, {files_skipped} skipped by gate")
    return {**state, "expanded_hits": expanded_hits, "phase": "expanded"}


# ── Node 6: judge_expanded_node ────────────────────────────────────────────────

def judge_expanded_node(state: SearchAgentState) -> SearchAgentState:
    """Judge the expanded hits."""
    expanded = state["expanded_hits"]
    query = state["normalized_query"]

    print(f"\n[5b/6] Judging {len(expanded)} expanded candidates...")

    llm = get_llm()
    new_accepted, new_rejected = _judge_hits(expanded, query, llm)

    print(f"  New accepted: {len(new_accepted)}  New rejected: {len(new_rejected)}")

    combined_accepted = state["accepted"] + new_accepted
    combined_rejected = state["rejected"] + new_rejected

    return {
        **state,
        "accepted": combined_accepted,
        "rejected": combined_rejected,
    }


# ── Node 7: format_node ────────────────────────────────────────────────────────

def format_node(state: SearchAgentState) -> SearchAgentState:
    """Expand accepted hits via cluster membership, deduplicate, build final results."""
    accepted = state["accepted"]
    query = state["normalized_query"]

    print(f"\n[6/6] Formatting results ({len(accepted)} direct matches)...")

    # Load cluster lookup
    cluster_lookup: dict[tuple[str, int], str] = {}   # (file, row_id) → cluster_id
    cluster_members: dict[str, list[tuple[str, int]]] = {}  # cluster_id → [(file, row_id)]

    if CLUSTERS_CSV.exists():
        clusters_df = pd.read_csv(CLUSTERS_CSV)
        for _, row in clusters_df.iterrows():
            key = (str(row["file"]), int(row["row_id"]))
            cid = str(row["cluster_id"])
            cluster_lookup[key] = cid
            cluster_members.setdefault(cid, []).append(key)

    supplier_data = state["supplier_data"]

    def get_product_info(fname: str, rid: int) -> tuple[str, float]:
        df = supplier_data.get(fname)
        if df is None:
            return ("", 0.0)
        row = df[df["row_id"] == rid]
        if row.empty:
            return ("", 0.0)
        r = row.iloc[0]
        return (str(r["product_name"]), float(r["price_rub"]) if pd.notna(r["price_rub"]) else 0.0)

    # Build result set
    seen: set[tuple[str, int]] = set()
    final_results: list[ResultItem] = []

    # 1. Direct matches
    for hit in accepted:
        key = (hit.file, hit.row_id)
        if key in seen:
            continue
        seen.add(key)
        cid = cluster_lookup.get(key, "")
        final_results.append(ResultItem(
            file=hit.file,
            row_id=hit.row_id,
            product_name=hit.product_name,
            price_rub=hit.price_rub,
            source="direct",
            cluster_id=cid,
        ))

    # 2. Cluster expansions
    for item in list(final_results):  # iterate over direct matches only
        if not item.cluster_id:
            continue
        for (fname, rid) in cluster_members.get(item.cluster_id, []):
            key = (fname, rid)
            if key in seen:
                continue
            seen.add(key)
            pname, price = get_product_info(fname, rid)
            final_results.append(ResultItem(
                file=fname,
                row_id=rid,
                product_name=pname,
                price_rub=price,
                source="cluster",
                cluster_id=item.cluster_id,
            ))

    # ── Print formatted output ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f'=== Результаты поиска: "{query}" ===')
    print(f"Найдено {len(final_results)} товаров:")

    # Group by source type
    direct = [r for r in final_results if r.source == "direct"]
    cluster_exp = [r for r in final_results if r.source == "cluster"]

    if direct:
        print("\n[ПРЯМЫЕ СОВПАДЕНИЯ]")
        for r in direct:
            supplier_tag = r.file.split("_")[1] if "_" in r.file else r.file
            cluster_tag = f"  (кластер {r.cluster_id})" if r.cluster_id else ""
            print(f"  {supplier_tag}:{r.row_id:4d}  {r.product_name[:65]}  — {r.price_rub:,.0f} руб.{cluster_tag}")

    if cluster_exp:
        # Group by cluster_id
        by_cluster: dict[str, list[ResultItem]] = {}
        for r in cluster_exp:
            by_cluster.setdefault(r.cluster_id, []).append(r)
        for cid, items in by_cluster.items():
            print(f"\n[ИЗ КЛАСТЕРА {cid}]")
            for r in items:
                supplier_tag = r.file.split("_")[1] if "_" in r.file else r.file
                print(f"  {supplier_tag}:{r.row_id:4d}  {r.product_name[:65]}  — {r.price_rub:,.0f} руб.")

    if not final_results:
        print("\n  Ничего не найдено.")

    print("=" * 70)

    return {**state, "final_results": final_results}


# ── Build graph ────────────────────────────────────────────────────────────────

def build_search_graph() -> Any:
    graph = StateGraph(SearchAgentState)

    graph.add_node("clarification_node", clarification_node)
    graph.add_node("normalize_node", normalize_node)
    graph.add_node("search_node", search_node)
    graph.add_node("judge_node", judge_node)
    graph.add_node("check_count_node", check_count_node)
    graph.add_node("expand_node", expand_node)
    graph.add_node("judge_expanded_node", judge_expanded_node)
    graph.add_node("format_node", format_node)

    graph.add_edge(START, "clarification_node")
    graph.add_edge("clarification_node", "normalize_node")
    graph.add_edge("normalize_node", "search_node")
    graph.add_edge("search_node", "judge_node")
    graph.add_edge("judge_node", "check_count_node")
    graph.add_conditional_edges(
        "check_count_node",
        route_after_check,
        {
            "expand": "expand_node",
            "format": "format_node",
        },
    )
    graph.add_edge("expand_node", "judge_expanded_node")
    graph.add_edge("judge_expanded_node", "format_node")
    graph.add_edge("format_node", END)

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

def run_search(
    query: str,
    supplier_data: dict | None = None,
    embedding_data: dict | None = None,
) -> list[ResultItem]:
    """Run the search agent for a single query.

    Args:
        query: Free-text user query.
        supplier_data: Pre-loaded supplier DataFrames (loaded if None).
        embedding_data: Pre-computed embeddings (loaded from cache if None).

    Returns:
        List of ResultItem objects found.
    """
    if supplier_data is None:
        print("Loading supplier data...")
        supplier_data = load_all_suppliers()

    if embedding_data is None:
        print("Loading embeddings from cache...")
        embedding_data = compute_embeddings(supplier_data)

    initial_state: SearchAgentState = {
        "raw_query": query,
        "clarification_answer": "",
        "normalized_query": "",
        "supplier_data": supplier_data,
        "embedding_data": embedding_data,
        "top15_per_file": {},
        "accepted": [],
        "rejected": [],
        "outliers": [],
        "expanded_hits": [],
        "final_results": [],
        "phase": "initial",
    }

    app = build_search_graph()
    final_state = app.invoke(initial_state)
    return final_state["final_results"]
