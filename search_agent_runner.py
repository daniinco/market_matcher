"""CLI entry point for the LangGraph product search agent.

Usage:
  # Single-shot mode:
  .venv/bin/python search_agent_runner.py "двухспальная кровать"
  .venv/bin/python search_agent_runner.py "Книжный шкаф Лофт 100×25×80 черный"

  # Interactive REPL mode (no arguments):
  .venv/bin/python search_agent_runner.py
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

    # ── Single-shot or interactive ─────────────────────────────────────────────
    if len(sys.argv) > 1:
        # Single-shot: query passed as CLI argument
        query = " ".join(sys.argv[1:])
        run_search(query, supplier_data=supplier_data, embedding_data=embedding_data)
    else:
        # Interactive REPL
        print("\nВведите запрос для поиска товара (или 'exit' для выхода):")
        print("-" * 60)
        while True:
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
