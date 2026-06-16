"""Run product matching pipeline with GigaChatEmbeddings-based candidate generation.

Steps:
  1. Load supplier data (all rows per supplier)
  2. Compute/load GigaChatEmbeddings (output/embeddings_cache.npz)
  3. Generate candidates: price window ±30% + cosine similarity ≥ 0.89
  4. Judge candidates with GigaChat-2-Pro (checkpoint/resume)
  5. Build clusters via Union-Find
  6. Save outputs to output/

Run again after interruption — resumes from checkpoint automatically.
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, "src")

from load_data import load_all_suppliers
from embeddings import compute_embeddings
from candidates import generate_candidates, COSINE_MIN
from judge import get_llm, judge_pair, JudgeResult
from clusters import build_clusters, save_outputs

MAX_ROWS   = None   # use all rows
SAVE_EVERY = 25
CHECKPOINT = Path("output/checkpoint_emb.jsonl")
CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print(f"PRODUCT MATCHING PIPELINE  (embedding cosine ≥ {COSINE_MIN})")
print("=" * 60)

# 1. Load data
print("\n[1/5] Loading supplier data (all rows)...")
data = load_all_suppliers(max_rows=MAX_ROWS)
total_rows = sum(len(df) for df in data.values())
print(f"Total rows loaded: {total_rows}")

# 2. Compute or load embeddings (auto-detects cache size mismatch)
print("\n[2/5] Computing/loading GigaChatEmbeddings...")
t0 = time.time()
embedding_data = compute_embeddings(data, force_recompute=False)   # set True to rebuild cache
print(f"Embeddings ready in {time.time()-t0:.1f}s")
for fname, (vecs, ids) in embedding_data.items():
    print(f"  {fname.split('_')[1]}: {vecs.shape[0]} vectors, dim={vecs.shape[1]}")

# 3. Generate candidates (cross-supplier only, embedding-based)
print("\n[3/5] Generating candidates (embedding cosine similarity)...")
t0 = time.time()
cands = generate_candidates(data, include_intra=False, embedding_data=embedding_data)
print(f"Done in {time.time()-t0:.1f}s")

if not cands:
    print("No candidates found — exiting.")
    sys.exit(1)

# Show cosine distribution of candidates
cosines = [c.token_jaccard for c in cands]
print(f"Cosine stats: min={min(cosines):.3f}  max={max(cosines):.3f}  mean={sum(cosines)/len(cosines):.3f}")

# 4. Load checkpoint if exists
done_keys: set[tuple] = set()
results: list[JudgeResult] = []

if CHECKPOINT.exists():
    with open(CHECKPOINT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            r = JudgeResult(
                left_file=d["left_file"],
                left_row_id=d["left_row_id"],
                right_file=d["right_file"],
                right_row_id=d["right_row_id"],
                label=d["label"],
                confidence=d["confidence"],
                reason=d["reason"],
                intra_supplier=d.get("intra_supplier", False),
            )
            results.append(r)
            done_keys.add((d["left_file"], d["left_row_id"], d["right_file"], d["right_row_id"]))
    print(f"\nResuming from checkpoint: {len(results)} pairs already done")

remaining = [
    c for c in cands
    if (c.left_file, c.left_row_id, c.right_file, c.right_row_id) not in done_keys
]
print(f"Remaining to judge: {len(remaining)}")

# 5. Judge remaining candidates
print(f"\n[4/5] Judging {len(remaining)} candidates with GigaChat...")
llm = get_llm()
total = len(cands)
done_so_far = len(results)

checkpoint_fh = open(CHECKPOINT, "a", encoding="utf-8")

try:
    for i, pair in enumerate(remaining):
        global_idx = done_so_far + i + 1
        result = judge_pair(pair, llm)
        results.append(result)

        icon = {"match": "✓", "non_match": "✗", "uncertain": "?"}.get(result.label, "?")
        extra = f"  — {result.reason[:60]}" if result.label != "non_match" else ""
        print(
            f"  [{global_idx}/{total}] {icon} {result.label} "
            f"(conf={result.confidence:.2f}  cosine={pair.token_jaccard:.3f}) "
            f"{pair.left_file.split('_')[1]}:{pair.left_row_id} × "
            f"{pair.right_file.split('_')[1]}:{pair.right_row_id}"
            + extra
        )

        checkpoint_fh.write(json.dumps({
            "left_file": result.left_file,
            "left_row_id": result.left_row_id,
            "right_file": result.right_file,
            "right_row_id": result.right_row_id,
            "label": result.label,
            "confidence": result.confidence,
            "reason": result.reason,
            "intra_supplier": result.intra_supplier,
        }, ensure_ascii=False) + "\n")

        if (i + 1) % SAVE_EVERY == 0:
            checkpoint_fh.flush()
            match_so_far = sum(1 for r in results if r.label == "match")
            print(f"\n  --- Checkpoint ({global_idx}/{total} done, {match_so_far} matches) ---\n")

finally:
    checkpoint_fh.flush()
    checkpoint_fh.close()

# 6. Build clusters and save
match_count     = sum(1 for r in results if r.label == "match")
non_match_count = sum(1 for r in results if r.label == "non_match")
uncertain_count = sum(1 for r in results if r.label == "uncertain")
print(f"\nFinal results: {match_count} match  {non_match_count} non_match  {uncertain_count} uncertain")

print("\n[5/5] Building clusters and saving outputs...")
pairs_df, clusters_df = build_clusters(results)
save_outputs(pairs_df, clusters_df)

print("\n=== Matches found ===")
matches = pairs_df[pairs_df["label"] == "match"]
print(f"Total matches: {len(matches)}")
for _, row in matches.iterrows():
    print(
        f"  [{row.left_file.split('_')[1]}:{row.left_row_id}] × "
        f"[{row.right_file.split('_')[1]}:{row.right_row_id}]  "
        f"conf={row.confidence:.2f}  {str(row.reason)[:80]}"
    )

print("\n=== Clusters ===")
if not clusters_df.empty:
    for cid, group in clusters_df.groupby("cluster_id"):
        members = [f"{r.file.split('_')[1]}:{r.row_id}" for _, r in group.iterrows()]
        print(f"  {cid}: {', '.join(members)}")
else:
    print("  No clusters formed.")

print("\nDone. Outputs saved to output/")
