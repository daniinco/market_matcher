"""Module 6: LangGraph agent orchestrating the full product matching pipeline."""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

# Ensure src/ is on the path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from candidates import CandidatePair, generate_candidates
from clusters import build_clusters, save_outputs
from evaluate import run_full_evaluation
from judge import JudgeResult, get_llm, judge_pair
from load_data import load_all_suppliers

warnings.filterwarnings("ignore", message="Unverified HTTPS request")
load_dotenv(override=True)

OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ── LangGraph State ────────────────────────────────────────────────────────────

class MatcherState(TypedDict):
    supplier_data: dict                  # dict[str, pd.DataFrame]
    candidates: list[CandidatePair]      # all candidate pairs
    current_index: int                   # loop pointer
    results: list[JudgeResult]           # judged results so far
    status: str                          # "running" | "done"


# ── Graph nodes ────────────────────────────────────────────────────────────────

def load_node(state: MatcherState) -> MatcherState:
    """Node 1: Load all supplier CSV files."""
    print("\n[1/4] Loading supplier data...")
    supplier_data = load_all_suppliers()
    return {**state, "supplier_data": supplier_data}


def generate_node(state: MatcherState) -> MatcherState:
    """Node 2: Generate candidate pairs using price-window + token-Jaccard blocking."""
    print("\n[2/4] Generating candidate pairs...")
    candidates = generate_candidates(state["supplier_data"], include_intra=True)
    return {**state, "candidates": candidates, "current_index": 0, "results": []}


def judge_node(state: MatcherState) -> MatcherState:
    """Node 3: Judge the current candidate pair with GigaChat."""
    idx = state["current_index"]
    candidates = state["candidates"]
    pair = candidates[idx]

    total = len(candidates)
    print(
        f"  [{idx+1}/{total}] Judging: "
        f"{pair.left_file.split('_')[1]}:{pair.left_row_id} × "
        f"{pair.right_file.split('_')[1]}:{pair.right_row_id} "
        f"(price_diff={pair.price_diff_pct}%, jaccard={pair.token_jaccard})"
    )

    llm = get_llm()
    result = judge_pair(pair, llm)

    icon = {"match": "✓", "non_match": "✗", "uncertain": "?"}.get(result.label, "?")
    print(f"    {icon} {result.label} (conf={result.confidence:.2f}): {result.reason[:80]}")

    new_results = state["results"] + [result]
    new_index = idx + 1
    new_status = "running" if new_index < total else "done"

    return {**state, "results": new_results, "current_index": new_index, "status": new_status}


def aggregate_node(state: MatcherState) -> MatcherState:
    """Node 4: Build clusters from matched pairs and save outputs."""
    print("\n[3/4] Building clusters...")
    results = state["results"]
    pairs_df, clusters_df = build_clusters(results)

    match_count = sum(1 for r in results if r.label == "match")
    non_match_count = sum(1 for r in results if r.label == "non_match")
    uncertain_count = sum(1 for r in results if r.label == "uncertain")
    print(f"  Results: {match_count} match, {non_match_count} non_match, {uncertain_count} uncertain")
    print(f"  Clusters formed: {clusters_df['cluster_id'].nunique() if not clusters_df.empty else 0}")

    print("\n[4/4] Saving outputs...")
    save_outputs(pairs_df, clusters_df, OUTPUT_DIR)

    return {**state, "status": "done"}


# ── Conditional edge ───────────────────────────────────────────────────────────

def should_continue(state: MatcherState) -> str:
    """Route back to judge_node while there are candidates left, else aggregate."""
    if state["current_index"] < len(state["candidates"]):
        return "judge_node"
    return "aggregate_node"


# ── Build graph ────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(MatcherState)

    graph.add_node("load_node", load_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("judge_node", judge_node)
    graph.add_node("aggregate_node", aggregate_node)

    graph.add_edge(START, "load_node")
    graph.add_edge("load_node", "generate_node")
    graph.add_edge("generate_node", "judge_node")
    graph.add_conditional_edges("judge_node", should_continue, {
        "judge_node": "judge_node",
        "aggregate_node": "aggregate_node",
    })
    graph.add_edge("aggregate_node", END)

    return graph.compile()


# ── QA single-scenario runner ──────────────────────────────────────────────────

def run_qa_scenario(case_id: str) -> None:
    """Run a single QA scenario by case_id and print result."""
    import json
    from pathlib import Path as P

    qa_path = P(__file__).parent.parent / "data" / "qa" / "qa_matching.jsonl"
    scenarios = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    scenario = next((s for s in scenarios if s["case_id"] == case_id), None)
    if scenario is None:
        print(f"QA case {case_id!r} not found.")
        return

    print(f"\nQA Scenario: {case_id} ({scenario['scenario_type']})")
    print(f"Dialog: {scenario['dialog'][0]['content']}")
    print(f"Expected outcome: {scenario['expected_outcome_type']}")

    # Load data and generate candidates for the specific context
    from candidates import get_candidates_for_pair
    from evaluate import evaluate_qa

    supplier_data = load_all_suppliers()
    ctx = scenario["input_context"]

    # For simple scenarios, judge the specific pair
    if "left_file" in ctx and "left_row_id" in ctx:
        right_file = ctx.get("right_file") or (ctx.get("candidate_files", [None])[0])
        if right_file:
            cands = get_candidates_for_pair(
                supplier_data,
                ctx["left_file"],
                ctx["left_row_id"],
                right_file,
            )
            print(f"\nCandidates found: {len(cands)}")
            if cands:
                llm = get_llm()
                from judge import judge_pairs_batch
                results = judge_pairs_batch(cands, llm=llm)
                from clusters import build_clusters, save_outputs
                pairs_df, clusters_df = build_clusters(results)
                save_outputs(pairs_df, clusters_df, OUTPUT_DIR)
                qa_results = evaluate_qa(pairs_df, clusters_df)
                for r in qa_results:
                    if r["case_id"] == case_id:
                        icon = "PASS" if r["passed"] else "FAIL"
                        print(f"\n[{icon}] {r['details']}")
            else:
                print("No candidates found for this pair.")
    else:
        print("Complex scenario — run full pipeline first, then evaluate.")


# ── Main entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Product Matching Pipeline")
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation after matching"
    )
    parser.add_argument(
        "--qa", metavar="CASE_ID",
        help="Run a single QA scenario (e.g. Q-001)"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only run evaluation on existing output files"
    )
    args = parser.parse_args()

    if args.eval_only:
        print("Running evaluation on existing outputs...")
        report = run_full_evaluation(output_dir=OUTPUT_DIR)
        print(report)
        return

    if args.qa:
        run_qa_scenario(args.qa)
        return

    # Full pipeline
    print("=" * 60)
    print("PRODUCT MATCHING PIPELINE")
    print("=" * 60)

    initial_state: MatcherState = {
        "supplier_data": {},
        "candidates": [],
        "current_index": 0,
        "results": [],
        "status": "running",
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    print(f"\nPipeline complete. Status: {final_state['status']}")
    print(f"Total pairs judged: {len(final_state['results'])}")

    if args.evaluate:
        print("\n" + "=" * 60)
        print("RUNNING EVALUATION")
        print("=" * 60)
        report = run_full_evaluation(output_dir=OUTPUT_DIR)
        print(report)


if __name__ == "__main__":
    main()
