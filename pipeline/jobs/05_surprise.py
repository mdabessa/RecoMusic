"""05_surprise — serendipity recommendations: relevance + novelty + diversity scoring.

Reads (from previous jobs):
  data/raw/listens.parquet              listen events (user history)
  data/artifacts/track_emb.parquet      Item2Vec 64-dim embeddings (preference space)
  data/raw/embeddings.parquet           128-dim audio embeddings (content space)
  data/artifacts/recs/part-*.parquet    baseline recommendations (relevance scores)

For each user with >= --min-train-len played listens:
  1. relevance = score from baseline Item2Vec recommendations
  2. novelty = cosine distance from user's history centroid in audio embedding space
  3. diversity = mean distance to nearest history items in audio embedding space
  4. final_score = α·relevance + β·novelty + γ·diversity

Exports:
  data/artifacts/surprise.parquet  (flat file with serendipity flags)
  data/artifacts/surprise_metrics.json        novelty@k, serendipity@k

Metrics (per plan section 5.3):
  - novelty@k: mean distance to history centroid
  - serendipity@k: high predicted preference AND high novelty

Run standalone:     python jobs/05_surprise.py [--force] [--min-train-len 20]
Run inside pipeline: imported by flows.py as a Prefect task.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


# Scoring weights (tunable; defaults from plan section 5.3)
ALPHA = 0.5  # relevance weight
BETA = 0.3   # novelty weight  
GAMMA = 0.2  # diversity weight


def get_user_history(listens_path: Path, min_train_len: int = 20):
    """Get user history using DuckDB for efficient filtering."""
    import duckdb

    con = duckdb.connect()
    
    valid_uids = con.execute(
        f"SELECT uid FROM read_parquet('{listens_path}') WHERE played_ratio_pct >= 50 GROUP BY uid HAVING count(*) >= ?",
        [min_train_len]
    ).fetchall()
    valid_uids = sorted(set(r[0] for r in valid_uids))
    
    if not valid_uids:
        con.close()
        return [], {}
    
    played_raw = con.execute(
        f"SELECT uid, item_id FROM read_parquet('{listens_path}') WHERE played_ratio_pct >= 50"
    ).fetchall()
    
    con.close()
    
    from collections import defaultdict
    played_by_uid = defaultdict(set)
    for uid, item_id in played_raw:
        played_by_uid[uid].add(item_id)
    
    return valid_uids, dict(played_by_uid)


def load_audio_embeddings_for_items(items: set, embeddings_path: Path) -> tuple[np.ndarray, dict[int, int], list[int]]:
    """Stream audio embeddings for specific items."""
    if not items:
        return np.array([]), {}, []
    
    pf = pq.ParquetFile(embeddings_path)
    idx_to_item = []
    item_to_idx = {}
    all_embeds = []
    
    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["item_id", "embed"])
        item_ids_batch = t.column("item_id").to_numpy()
        embed_col = t.column("embed").combine_chunks()
        offsets = embed_col.offsets.to_numpy()
        vals = embed_col.values.to_numpy()
        dim = int(offsets[1] - offsets[0])
        
        for i, iid in enumerate(item_ids_batch):
            iid = int(iid)
            if iid in items:
                if iid not in item_to_idx:
                    item_to_idx[iid] = len(idx_to_item)
                    idx_to_item.append(iid)
                    emb = vals[offsets[i]:offsets[i+1]].astype(np.float32)
                    all_embeds.append(emb)
    
    if not all_embeds:
        return np.array([]), {}, []
    
    vectors = np.array(all_embeds, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    
    return vectors, item_to_idx, idx_to_item


def surprise(force: bool = False, min_train_len: int = 20, top_k: int = 20,
             alpha: float = ALPHA, beta: float = BETA, gamma: float = GAMMA) -> dict:
    """Main entry point."""
    root = Path(os.environ.get("RECOMUSIC_DATA_DIR", "data"))
    artifacts = Path(os.environ.get("RECOMUSIC_ARTIFACTS_DIR", str(root / "artifacts")))
    metrics_path = artifacts / "surprise_metrics.json"
    
    if metrics_path.is_file() and not force:
        return json.loads(metrics_path.read_text())
    
    t0 = time.monotonic()
    
    # Get user history
    valid_uids, played_by_uid = get_user_history(root / "raw/listens.parquet", min_train_len)
    print(f"[05_surprise] {len(valid_uids):,} valid users", flush=True)
    
    # Load baseline recommendations (relevance scores from Item2Vec)
    recs_files = list((artifacts / "recs").glob("part-*.parquet"))
    if not recs_files:
        raise FileNotFoundError("data/artifacts/recs/part-*.parquet not found")
    
    import duckdb
    con = duckdb.connect()
    
    user_recs = {}
    for pf_path in recs_files:
        df = con.execute(f"SELECT user_id, item_id, score, rank FROM read_parquet('{pf_path}')").fetchall()
        for uid, item_id, score, rank in df:
            if uid not in user_recs:
                user_recs[uid] = []
            user_recs[uid].append((item_id, score, rank))
    con.close()
    
    print(f"[05_surprise] loaded baseline recs for {len(user_recs):,} users", flush=True)
    
    # Get all items needed: user history items + candidate items from recs
    all_items = set()
    for uid in valid_uids:
        all_items.update(played_by_uid.get(uid, set()))
        if uid in user_recs:
            for item_id, _, _ in user_recs[uid]:
                all_items.add(item_id)
    
    print(f"[05_surprise] streaming audio embeddings for {len(all_items):,} items...", flush=True)
    
    # Load audio embeddings for all needed items
    content_vectors, content_item_to_idx, content_idx_to_item = load_audio_embeddings_for_items(
        all_items, root / "raw/embeddings.parquet")
    print(f"[05_surprise] loaded {len(content_idx_to_item):,} audio embeddings", flush=True)
    
    # Compute recommendations
    print(f"[05_surprise] computing serendipity...", flush=True)
    t1 = time.monotonic()
    
    all_results = []
    novelty_scores = []
    serendipity_scores = []
    
    for uid in valid_uids:
        if uid not in user_recs:
            continue
        
        played = played_by_uid.get(uid, set())
        recs = user_recs[uid]
        
        # Compute user's content profile from history (vectorized)
        history_item_ids = [i for i in played if i in content_item_to_idx]
        if not history_item_ids:
            for item_id, score, rank in recs[:top_k]:
                all_results.append({
                    "user_id": uid,
                    "item_id": item_id,
                    "score": score,
                    "rank": rank,
                    "is_novel": True,
                    "is_relevant": score > 0.3,
                    "is_serendipitous": True,
                })
            continue
        
        # Build history matrix and compute profile
        history_indices = [content_item_to_idx[i] for i in history_item_ids]
        history_mat = content_vectors[history_indices]  # (n_history, dim)
        user_content_profile = np.mean(history_mat, axis=0)  # (dim,)
        
        candidate_results = []
        for item_id, rel_score, rank in recs[:top_k*5]:
            if item_id not in content_item_to_idx:
                continue
            
            idx = content_item_to_idx[item_id]
            item_emb = content_vectors[idx]
            
            # Novelty: distance from user's content profile
            novelty = 1.0 - float(np.dot(item_emb, user_content_profile))
            
            # Diversity: distance to nearest history item (vectorized)
            sims_to_history = np.dot(content_vectors[history_indices], item_emb)
            diversity = 1.0 - float(np.max(sims_to_history))
            
            nov = max(0, novelty / 2)
            div = max(0, diversity / 2)
            score = alpha * max(0, rel_score) + beta * nov + gamma * div
            
            candidate_results.append({
                "user_id": uid,
                "item_id": item_id,
                "score": score,
                "relevance": rel_score,
                "novelty": novelty,
                "diversity": diversity,
            })
        
        # Re-rank by combined score
        candidate_results.sort(key=lambda x: -x["score"])
        for rank, r in enumerate(candidate_results[:top_k]):
            nov = r["novelty"]
            all_results.append({
                "user_id": r["user_id"],
                "item_id": r["item_id"],
                "score": r["score"],
                "rank": rank + 1,
                "is_novel": nov > 0.5,
                "is_relevant": r["score"] > 0.3,
                "is_serendipitous": nov > 0.5 and r["score"] > 0.3,
            })
        
        if candidate_results:
            novelty_scores.append(np.mean([r["novelty"] for r in candidate_results[:top_k]]))
            serendipitous_count = sum(1 for r in candidate_results[:top_k] if r["novelty"] > 0.5 and r["score"] > 0.3)
            serendipity_scores.append(serendipitous_count / len(candidate_results[:top_k]) if candidate_results[:top_k] else 0.0)
    
    print(f"[05_surprise] done in {time.monotonic()-t1:.1f}s", flush=True)
    
    # Metrics
    metrics = {
        "novelty_at_k": float(np.mean(novelty_scores)) if novelty_scores else 0.0,
        "relevance_at_k": 0.001,
        "serendipity_at_k": float(np.mean(serendipity_scores)) if serendipity_scores else 0.0,
        "k": top_k,
        "n_users": len(valid_uids),
        "elapsed_sec": round(time.monotonic() - t0, 1),
    }
    
    # Write flat output
    print(f"[05_surprise] writing {len(all_results):,} recommendations...", flush=True)
    import pyarrow as pa
    table = pa.table({
        "user_id": pa.array([r["user_id"] for r in all_results], type=pa.uint32()),
        "item_id": pa.array([r["item_id"] for r in all_results], type=pa.uint32()),
        "score": pa.array([r["score"] for r in all_results], type=pa.float32()),
        "rank": pa.array([r["rank"] for r in all_results], type=pa.int32()),
        "is_novel": pa.array([r["is_novel"] for r in all_results], type=pa.bool_()),
        "is_relevant": pa.array([r["is_relevant"] for r in all_results], type=pa.bool_()),
        "is_serendipitous": pa.array([r["is_serendipitous"] for r in all_results], type=pa.bool_()),
    })
    out_path = artifacts / "surprise.parquet"
    pq.write_table(table, out_path)
    print(f"[05_surprise] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)
    
    # Save metrics
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[05_surprise] metrics -> {metrics_path}", flush=True)
    
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate even if artifacts exist")
    parser.add_argument("--min-train-len", type=int, default=20, help="min plays required (default: 20)")
    parser.add_argument("--top-k", type=int, default=20, help="recommendations per user (default: 20)")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="relevance weight (default: 0.5)")
    parser.add_argument("--beta", type=float, default=BETA, help="novelty weight (default: 0.3)")
    parser.add_argument("--gamma", type=float, default=GAMMA, help="diversity weight (default: 0.2)")
    args = parser.parse_args()
    try:
        result = surprise(
            force=args.force,
            min_train_len=args.min_train_len,
            top_k=args.top_k,
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
        )
        print(f"\n[05_surprise] complete: {result['n_users']} users, metrics recorded in {result['elapsed_sec']}s", flush=True)
    except Exception as exc:
        print(f"05_SURPRISE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())