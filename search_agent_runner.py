"""CLI entry point for the LangGraph product search agent.

Usage:
  # Pass initial query as argument (stdin stays open for clarification answers):
  .venv/bin/python search_agent_runner.py "стол"
  .venv/bin/python search_agent_runner.py "двухспальная кровать"

  # Interactive REPL mode (no arguments — agent asks for query interactively):
  .venv/bin/python search_agent_runner.py

In both modes stdin remains open, so when the clarification_node asks a
follow-up question the user can type the answer directly in the terminal.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv(override=True)

from embeddings import compute_embeddings
from load_data import load_all_suppliers
from search_agent import run_search


def main() -> None:
    # ── Pre-load data and embeddings once ─────────────────────────────────────
    print("=" * 60)
    print("PRODUCT SEARCH AGENT  (GigaChat + GigaChatEmbeddings)")
    print("=" * 60)

    print("\nLoading supplier data...")
    supplier_data = load_all_suppliers()
    total_rows = sum(len(df) for df in supplier_data.values())
    print(f"  {len(supplier_data)} suppliers, {total_rows} rows total")

    print("\nLoading embeddings from cache...")
    embedding_data = compute_embeddings(supplier_data)
    total_vecs = sum(v[0].shape[0] for v in embedding_data.values())
    print(f"  {total_vecs} vectors loaded")

    # ── Determine first query ──────────────────────────────────────────────────
    # If a query was passed as CLI argument, use it for the first search.
    # Either way, stdin stays open so clarification_node can call input().
    first_query: str | None = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None

    print("\nВведите запрос для поиска товара (или 'exit' для выхода):")
    print("-" * 60)

    while True:
        if first_query is not None:
            # Use the CLI argument as the first query without prompting
            query = first_query
            first_query = None          # only use it once
            print(f"\n> {query}")       # echo so the user sees what was submitted
        else:
            try:
                query = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nВыход.")
                break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "выход", "q"):
            print("Выход.")
            break

        run_search(query, supplier_data=supplier_data, embedding_data=embedding_data)


if __name__ == "__main__":
    main()
