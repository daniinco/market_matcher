"""Module 4: Build clusters from matched pairs using Union-Find."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from judge import JudgeResult

OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ── Union-Find ─────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self) -> None:
        self._parent: dict[tuple, tuple] = {}
        self._rank: dict[tuple, int] = {}

    def find(self, x: tuple) -> tuple:
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # path compression
        return self._parent[x]

    def union(self, x: tuple, y: tuple) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        # Union by rank
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> dict[tuple, list[tuple]]:
        """Return dict mapping root -> list of members."""
        groups: dict[tuple, list[tuple]] = {}
        for node in self._parent:
            root = self.find(node)
            groups.setdefault(root, []).append(node)
        return groups


# ── Cluster builder ────────────────────────────────────────────────────────────

def build_clusters(
    results: list[JudgeResult],
    min_confidence: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build clusters from JudgeResult list.

    Args:
        results: list of JudgeResult from judge_pairs_batch()
        min_confidence: minimum confidence to include a match in clusters

    Returns:
        (pairs_df, clusters_df) — both as DataFrames ready to save
    """
    uf = UnionFind()

    # Register all nodes and union matched pairs
    for r in results:
        node_l = (r.left_file, r.left_row_id)
        node_r = (r.right_file, r.right_row_id)
        # Ensure both nodes exist in UF even if non_match
        uf.find(node_l)
        uf.find(node_r)
        if r.label == "match" and r.confidence >= min_confidence:
            uf.union(node_l, node_r)

    # Build clusters DataFrame — only clusters with >= 2 members
    components = uf.components()
    cluster_rows = []
    cluster_counter = 1
    for root, members in sorted(components.items()):
        if len(members) < 2:
            continue
        cid = f"C-{cluster_counter:04d}"
        cluster_counter += 1
        for file, row_id in sorted(members):
            cluster_rows.append({
                "cluster_id": cid,
                "file": file,
                "row_id": row_id,
            })

    clusters_df = pd.DataFrame(cluster_rows, columns=["cluster_id", "file", "row_id"])

    # Build pairs DataFrame — all judged pairs
    pairs_rows = [
        {
            "left_file": r.left_file,
            "left_row_id": r.left_row_id,
            "right_file": r.right_file,
            "right_row_id": r.right_row_id,
            "label": r.label,
            "confidence": round(r.confidence, 4),
            "reason": r.reason,
            "intra_supplier": r.intra_supplier,
        }
        for r in results
    ]
    pairs_df = pd.DataFrame(pairs_rows)

    return pairs_df, clusters_df


def save_outputs(
    pairs_df: pd.DataFrame,
    clusters_df: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Save pairs and clusters DataFrames to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = output_dir / "pairs_predicted.csv"
    clusters_path = output_dir / "clusters_predicted.csv"
    pairs_df.to_csv(pairs_path, index=False)
    clusters_df.to_csv(clusters_path, index=False)
    print(f"  Saved {len(pairs_df)} pairs → {pairs_path}")
    print(f"  Saved {len(clusters_df)} cluster rows ({clusters_df['cluster_id'].nunique()} clusters) → {clusters_path}")
    return pairs_path, clusters_path


if __name__ == "__main__":
    # Quick smoke test with dummy data
    dummy = [
        JudgeResult("S1.csv", 1, "S2.csv", 10, "match", 0.95, "same product"),
        JudgeResult("S1.csv", 1, "S3.csv", 20, "match", 0.88, "same product"),
        JudgeResult("S2.csv", 10, "S3.csv", 20, "match", 0.91, "same product"),
        JudgeResult("S1.csv", 2, "S2.csv", 11, "non_match", 0.92, "different model"),
        JudgeResult("S3.csv", 30, "S4.csv", 40, "uncertain", 0.45, "not enough info"),
    ]
    pairs_df, clusters_df = build_clusters(dummy)
    print("Pairs:")
    print(pairs_df.to_string(index=False))
    print("\nClusters:")
    print(clusters_df.to_string(index=False))
