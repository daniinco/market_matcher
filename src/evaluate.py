"""Module 5: Evaluate predicted pairs and clusters against gold-standard data and QA scenarios."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REFERENCE_DIR = Path(__file__).parent.parent / "data" / "reference"
QA_DIR = Path(__file__).parent.parent / "data" / "qa"
OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pair_key(left_file: str, left_row_id: int, right_file: str, right_row_id: int) -> frozenset:
    """Canonical key for a pair (order-independent)."""
    return frozenset([(left_file, int(left_row_id)), (right_file, int(right_row_id))])


def _load_gold_pairs() -> pd.DataFrame:
    path = REFERENCE_DIR / "pairs_labeled_train.csv"
    return pd.read_csv(path)


def _load_gold_clusters() -> pd.DataFrame:
    path = REFERENCE_DIR / "clusters_gold.csv"
    return pd.read_csv(path)


def _load_qa_scenarios() -> list[dict]:
    path = QA_DIR / "qa_matching.jsonl"
    scenarios = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


# ── Evaluation 1: Pairs vs gold ────────────────────────────────────────────────

def evaluate_pairs(predicted_df: pd.DataFrame) -> dict:
    """Compare predicted pairs against pairs_labeled_train.csv.

    Returns dict with precision, recall, F1 for 'match' class and confusion matrix.
    """
    gold_df = _load_gold_pairs()

    # Build lookup: pair_key -> gold_label
    gold_lookup: dict[frozenset, str] = {}
    for _, row in gold_df.iterrows():
        key = _pair_key(row["left_file"], row["left_row_id"], row["right_file"], row["right_row_id"])
        gold_lookup[key] = row["label"]

    # Build lookup: pair_key -> predicted_label
    pred_lookup: dict[frozenset, str] = {}
    for _, row in predicted_df.iterrows():
        key = _pair_key(row["left_file"], row["left_row_id"], row["right_file"], row["right_row_id"])
        pred_lookup[key] = row["label"]

    # Confusion matrix over gold pairs
    confusion: dict[str, dict[str, int]] = {
        "match": {"match": 0, "non_match": 0, "uncertain": 0, "missing": 0},
        "non_match": {"match": 0, "non_match": 0, "uncertain": 0, "missing": 0},
        "uncertain": {"match": 0, "non_match": 0, "uncertain": 0, "missing": 0},
    }

    for key, gold_label in gold_lookup.items():
        pred_label = pred_lookup.get(key, "missing")
        if gold_label in confusion:
            confusion[gold_label][pred_label] = confusion[gold_label].get(pred_label, 0) + 1

    # Precision / Recall / F1 for "match" class
    tp = confusion["match"]["match"]
    fp = sum(
        1 for key, pred in pred_lookup.items()
        if pred == "match" and gold_lookup.get(key, "non_match") != "match"
    )
    fn = confusion["match"]["non_match"] + confusion["match"]["uncertain"] + confusion["match"]["missing"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "confusion": confusion,
        "gold_pairs_total": len(gold_lookup),
        "predicted_pairs_total": len(pred_lookup),
    }


# ── Evaluation 2: Clusters vs gold ────────────────────────────────────────────

def evaluate_clusters(predicted_clusters_df: pd.DataFrame) -> dict:
    """Compare predicted clusters against clusters_gold.csv.

    Returns cluster-level precision, recall, F1 and per-cluster status.
    """
    gold_df = _load_gold_clusters()

    # Build gold cluster sets: cluster_id -> frozenset of (file, row_id)
    gold_clusters: dict[str, frozenset] = {}
    for cid, group in gold_df.groupby("cluster_id"):
        gold_clusters[cid] = frozenset(
            (row["file"], int(row["row_id"])) for _, row in group.iterrows()
        )

    # Build predicted cluster sets
    pred_clusters: dict[str, frozenset] = {}
    if not predicted_clusters_df.empty:
        for cid, group in predicted_clusters_df.groupby("cluster_id"):
            pred_clusters[cid] = frozenset(
                (row["file"], int(row["row_id"])) for _, row in group.iterrows()
            )

    # For each gold cluster, find best matching predicted cluster (by Jaccard)
    per_cluster = []
    matched_pred_ids: set[str] = set()

    for gold_id, gold_set in gold_clusters.items():
        best_jaccard = 0.0
        best_pred_id = None
        for pred_id, pred_set in pred_clusters.items():
            if pred_id in matched_pred_ids:
                continue
            j = len(gold_set & pred_set) / len(gold_set | pred_set) if (gold_set | pred_set) else 0.0
            if j > best_jaccard:
                best_jaccard = j
                best_pred_id = pred_id

        status = "missed"
        if best_jaccard >= 0.5:
            status = "correct" if best_jaccard >= 0.9 else "partial"
            if best_pred_id:
                matched_pred_ids.add(best_pred_id)

        per_cluster.append({
            "gold_cluster_id": gold_id,
            "best_pred_cluster_id": best_pred_id,
            "jaccard": round(best_jaccard, 4),
            "status": status,
            "gold_size": len(gold_set),
        })

    correct = sum(1 for c in per_cluster if c["status"] == "correct")
    partial = sum(1 for c in per_cluster if c["status"] == "partial")
    missed = sum(1 for c in per_cluster if c["status"] == "missed")
    spurious = len(pred_clusters) - len(matched_pred_ids)

    cluster_precision = correct / len(pred_clusters) if pred_clusters else 0.0
    cluster_recall = correct / len(gold_clusters) if gold_clusters else 0.0
    cluster_f1 = (
        2 * cluster_precision * cluster_recall / (cluster_precision + cluster_recall)
        if (cluster_precision + cluster_recall) > 0 else 0.0
    )

    return {
        "cluster_precision": round(cluster_precision, 4),
        "cluster_recall": round(cluster_recall, 4),
        "cluster_f1": round(cluster_f1, 4),
        "correct": correct,
        "partial": partial,
        "missed": missed,
        "spurious": spurious,
        "gold_clusters_total": len(gold_clusters),
        "predicted_clusters_total": len(pred_clusters),
        "per_cluster": per_cluster,
    }


# ── Evaluation 3: QA scenarios ─────────────────────────────────────────────────

def evaluate_qa(
    predicted_pairs_df: pd.DataFrame,
    predicted_clusters_df: pd.DataFrame,
) -> list[dict]:
    """Check each QA scenario against predicted outputs."""
    scenarios = _load_qa_scenarios()
    results = []

    pred_pair_lookup: dict[frozenset, dict] = {}
    for _, row in predicted_pairs_df.iterrows():
        key = _pair_key(row["left_file"], row["left_row_id"], row["right_file"], row["right_row_id"])
        pred_pair_lookup[key] = row.to_dict()

    for scenario in scenarios:
        case_id = scenario["case_id"]
        stype = scenario["scenario_type"]
        expected_type = scenario["expected_outcome_type"]
        expected = scenario["expected_result"]
        ctx = scenario["input_context"]

        passed = False
        details = ""

        if stype == "simple_match":
            # Q-001: expect pair S1:101 - S2:205 with match
            exp_pairs = expected.get("pairs", [])
            for ep in exp_pairs:
                key = _pair_key(ep["left_file"], ep["left_row_id"], ep["right_file"], ep["right_row_id"])
                pred = pred_pair_lookup.get(key)
                if pred and pred["label"] == "match":
                    passed = True
                    details = f"Found match with confidence={pred['confidence']:.2f}"
                else:
                    details = f"Pair not found or not labeled match (found: {pred['label'] if pred else 'missing'})"

        elif stype == "hard_match":
            # Q-002: expect S3:310 in candidate list
            exp_candidates = expected.get("candidates", [])
            for ec in exp_candidates:
                key = _pair_key(ctx["left_file"], ctx["left_row_id"], ec["file"], ec["row_id"])
                pred = pred_pair_lookup.get(key)
                if pred and pred["label"] in ("match", "uncertain"):
                    passed = True
                    details = f"Hard match found: {pred['label']} conf={pred['confidence']:.2f}"
                else:
                    details = f"Hard match not found (found: {pred['label'] if pred else 'missing'})"

        elif stype == "price_conflict":
            # Q-003: expect match despite price diff
            exp_pairs = expected.get("pairs", [])
            for ep in exp_pairs:
                key = _pair_key(ep["left_file"], ep["left_row_id"], ep["right_file"], ep["right_row_id"])
                pred = pred_pair_lookup.get(key)
                if pred and pred["label"] == "match":
                    passed = True
                    details = f"Price-conflict match found conf={pred['confidence']:.2f}"
                else:
                    details = f"Price-conflict match not found (found: {pred['label'] if pred else 'missing'})"

        elif stype == "duplicate_inside_supplier":
            # Q-004: expect rows 150+151 in same cluster
            exp_clusters = expected.get("clusters", [])
            for ec in exp_clusters:
                rows = ec.get("rows", [])
                fname = ctx.get("file", "")
                if not predicted_clusters_df.empty:
                    # Find cluster containing row 150
                    mask = (
                        (predicted_clusters_df["file"] == fname) &
                        (predicted_clusters_df["row_id"] == rows[0])
                    )
                    matching = predicted_clusters_df[mask]
                    if not matching.empty:
                        cid = matching.iloc[0]["cluster_id"]
                        cluster_rows = predicted_clusters_df[
                            predicted_clusters_df["cluster_id"] == cid
                        ]["row_id"].tolist()
                        if rows[1] in cluster_rows:
                            passed = True
                            details = f"Intra-supplier duplicate cluster found: {cid} rows={cluster_rows}"
                        else:
                            details = f"Row {rows[1]} not in same cluster as {rows[0]} (cluster {cid}: {cluster_rows})"
                    else:
                        details = f"Row {rows[0]} not in any cluster"
                else:
                    details = "No clusters predicted"

        elif stype == "uncertain_case":
            # Q-005: expect escalation (uncertain label)
            left_file = ctx["left_file"]
            left_row_id = ctx["left_row_id"]
            right_file = ctx["right_file"]
            # Check if any pair from left_row_id to right_file is uncertain
            uncertain_found = False
            for key, pred in pred_pair_lookup.items():
                nodes = list(key)
                files_rows = {(n[0], n[1]) for n in nodes}
                if (left_file, left_row_id) in files_rows:
                    if any(n[0] == right_file for n in nodes):
                        if pred["label"] == "uncertain":
                            uncertain_found = True
                            details = f"Correctly escalated: conf={pred['confidence']:.2f} reason={pred['reason'][:60]}"
                            break
            if not uncertain_found:
                details = "No uncertain escalation found for this pair"
            passed = uncertain_found

        elif stype == "cluster_validation":
            # Q-006: expect explanation that C-0001 is valid
            gold_df = _load_gold_clusters()
            cid = ctx.get("cluster_id", "C-0001")
            gold_members = gold_df[gold_df["cluster_id"] == cid]
            if not predicted_clusters_df.empty:
                # Find predicted cluster that overlaps most with gold C-0001
                gold_set = frozenset(
                    (row["file"], int(row["row_id"])) for _, row in gold_members.iterrows()
                )
                best_overlap = 0
                best_pred_cid = None
                for pcid, group in predicted_clusters_df.groupby("cluster_id"):
                    pred_set = frozenset(
                        (row["file"], int(row["row_id"])) for _, row in group.iterrows()
                    )
                    overlap = len(gold_set & pred_set)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_pred_cid = pcid
                if best_overlap >= len(gold_set) * 0.5:
                    passed = True
                    details = f"Cluster {cid} validated: predicted {best_pred_cid} overlaps {best_overlap}/{len(gold_set)} members"
                else:
                    details = f"Cluster {cid} not validated: best overlap={best_overlap}/{len(gold_set)}"
            else:
                details = "No clusters predicted"

        results.append({
            "case_id": case_id,
            "scenario_type": stype,
            "expected_outcome_type": expected_type,
            "passed": passed,
            "details": details,
        })

    return results


# ── Full evaluation report ─────────────────────────────────────────────────────

def run_full_evaluation(
    pairs_path: Path | None = None,
    clusters_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> str:
    """Run all evaluations and return a formatted report string."""
    if pairs_path is None:
        pairs_path = output_dir / "pairs_predicted.csv"
    if clusters_path is None:
        clusters_path = output_dir / "clusters_predicted.csv"

    if not pairs_path.exists():
        return f"ERROR: {pairs_path} not found. Run matcher.py first."

    pairs_df = pd.read_csv(pairs_path)
    clusters_df = pd.read_csv(clusters_path) if clusters_path.exists() else pd.DataFrame()

    lines = ["=" * 60, "PRODUCT MATCHING EVALUATION REPORT", "=" * 60]

    # 1. Pairs evaluation
    lines.append("\n## 1. Pairs vs Gold (pairs_labeled_train.csv)")
    pair_metrics = evaluate_pairs(pairs_df)
    lines.append(f"  Precision : {pair_metrics['precision']:.4f}")
    lines.append(f"  Recall    : {pair_metrics['recall']:.4f}")
    lines.append(f"  F1        : {pair_metrics['f1']:.4f}")
    lines.append(f"  TP={pair_metrics['tp']}  FP={pair_metrics['fp']}  FN={pair_metrics['fn']}")
    lines.append(f"  Gold pairs: {pair_metrics['gold_pairs_total']}  Predicted: {pair_metrics['predicted_pairs_total']}")
    lines.append("\n  Confusion matrix (gold_label → predicted_label):")
    for gold_label, counts in pair_metrics["confusion"].items():
        lines.append(f"    {gold_label:12s}: {counts}")

    # 2. Clusters evaluation
    lines.append("\n## 2. Clusters vs Gold (clusters_gold.csv)")
    cluster_metrics = evaluate_clusters(clusters_df)
    lines.append(f"  Cluster Precision : {cluster_metrics['cluster_precision']:.4f}")
    lines.append(f"  Cluster Recall    : {cluster_metrics['cluster_recall']:.4f}")
    lines.append(f"  Cluster F1        : {cluster_metrics['cluster_f1']:.4f}")
    lines.append(f"  Correct={cluster_metrics['correct']}  Partial={cluster_metrics['partial']}  Missed={cluster_metrics['missed']}  Spurious={cluster_metrics['spurious']}")
    lines.append(f"  Gold clusters: {cluster_metrics['gold_clusters_total']}  Predicted: {cluster_metrics['predicted_clusters_total']}")
    lines.append("\n  Per-cluster details:")
    for c in cluster_metrics["per_cluster"]:
        lines.append(
            f"    {c['gold_cluster_id']}: {c['status']:8s} "
            f"jaccard={c['jaccard']:.3f} "
            f"(gold_size={c['gold_size']}, pred={c['best_pred_cluster_id']})"
        )

    # 3. QA scenarios
    lines.append("\n## 3. QA Scenarios (qa_matching.jsonl)")
    qa_results = evaluate_qa(pairs_df, clusters_df)
    passed_count = sum(1 for r in qa_results if r["passed"])
    lines.append(f"  Passed: {passed_count}/{len(qa_results)}")
    for r in qa_results:
        icon = "PASS" if r["passed"] else "FAIL"
        lines.append(f"  [{icon}] {r['case_id']} ({r['scenario_type']}): {r['details']}")

    lines.append("\n" + "=" * 60)
    report = "\n".join(lines)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "evaluation_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"  Evaluation report saved → {report_path}")

    return report


if __name__ == "__main__":
    report = run_full_evaluation()
    print(report)
