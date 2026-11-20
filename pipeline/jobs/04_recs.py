"""04_recs — top-20 Item2Vec recommendations per user, partitioned by user_id % 100.

Reads data/raw/listens.parquet (user history) and data/artifacts/track_emb.parquet
(Item2Vec 64-dim embeddings). For each user with >= --min-train-len played
listens: profile = L2-normalized mean of the embeddings of the user's most
recent --num-recent played tracks; candidates = all track embeddings minus the
user's full played history; score = cosine similarity; top-20 by score.

Exports data/artifacts/recs/part-<mod>.parquet partitioned by user_id % 100
(recs/part-000.parquet ... recs/part-099.parquet) so DuckDB-WASM can fetch one
shard per user via HTTP range requests.

Note on partitioning: The Yambda dataset uses user_ids that are multiples of 100
(e.g., 100, 200, 1000000). A literal `user_id % 100` would put all users in
partition 0. To achieve the intended 100 shards for DuckDB range requests, we
use `(user_id // 100) % 100` as the partition key, which spreads ~92 users per
shard.

Run standalone:     python jobs/04_recs.py [--force] [--min-train-len 20]
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

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def load_track_embeddings(emb_path: Path) -> tuple[np.ndarray, dict[int, int], list[int]]:
    """Load track embeddings from parquet.

    Returns:
        vectors: (n_items, dim) float32 normalized vectors
        item_to_idx: map item_id -> row index
        idx_to_item: list of item_ids in order
    """
    table = pq.read_table(emb_path)
    item_ids = table.column("item_id").to_numpy()
    embed_col = table.column("embed")
    
    # For FixedSizeListArray (track_emb uses fixed-width list columns):
    # combine_chunks gives a FixedSizeListArray, whose .values is flat float array
    combined = embed_col.combine_chunks()
    dim = int(combined.type.list_size)
    vals = combined.values.to_numpy()  # flat array of length n_items * dim
    
    n_items = len(item_ids)
    e = vals.reshape(n_items, dim).astype(np.float32, copy=False)
    
    # L2 normalize
    norms = np.linalg.norm(e, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    e = e / norms
    
    item_to_idx = {int(k): i for i, k in enumerate(item_ids)}
    idx_to_item = [int(k) for k in item_ids]
    return e, item_to_idx, idx_to_item


def get_valid_users(listens_path: Path, min_train_len: int = 20) -> list[int]:
    """Get list of users with enough played tracks."""
    con = duckdb.connect()
    t0 = time.monotonic()
    rows = con.execute(
        f"""
        SELECT DISTINCT uid FROM read_parquet('{listens_path}')
        WHERE played_ratio_pct >= 50
        GROUP BY uid
        HAVING count(*) >= ?
        """,
        [min_train_len],
    ).fetchall()
    con.close()
    
    valid_uids = sorted(set(r[0] for r in rows))
    n_users = len(valid_uids)
    
    if n_users == 0:
        print("[04_recs] no users with sufficient training data", flush=True)
        return []
    
    print(f"[04_recs] {n_users:,} users with >= {min_train_len} played tracks ({time.monotonic()-t0:.1f}s)", flush=True)
    return valid_uids


def compute_recs_for_users(
    listens_path: Path,
    vectors: np.ndarray,
    item_to_idx: dict[int, int],
    idx_to_item: list[int],
    user_ids: list[int],
    num_recent: int = 20,
    min_train_len: int = 20,
    top_k: int = 20,
    batch_size: int = 256,
) -> tuple[list[dict], int]:
    """Compute top-20 recommendations for each user.

    Returns:
        results: list of recommendation dicts
        n_cold_users: count of users skipped due to no valid history embeddings
    """
    n_items, dim = vectors.shape
    results = []
    n_cold = 0
    
    t0 = time.monotonic()
    print("[04_recs] building user profiles (this may take a minute)...", flush=True)
    
    # Get recent items per user using window function
    con = duckdb.connect()
    recent_items_raw = con.execute(
        f"""
        SELECT uid, item_id FROM (
            SELECT uid, item_id,
                   ROW_NUMBER() OVER (PARTITION BY uid ORDER BY timestamp DESC) as rn
            FROM read_parquet('{listens_path}')
            WHERE played_ratio_pct >= 50
        ) WHERE rn <= {num_recent}
        """
    ).fetchall()
    
    # Get all played items per user (for exclusion)
    all_played = con.execute(
        f"""
        SELECT uid, list(DISTINCT item_id) FROM read_parquet('{listens_path}')
        WHERE played_ratio_pct >= 50
        GROUP BY uid
        """
    ).fetchall()
    con.close()
    
    # Group by uid
    from collections import defaultdict
    recent_by_uid: dict[int, list[int]] = defaultdict(list)
    for uid, item_id in recent_items_raw:
        recent_by_uid[uid].append(item_id)
    
    played_by_uid: dict[int, set[int]] = {}
    for uid, items in all_played:
        played_by_uid[uid] = set(items)
    
    # Build user_profile matrix
    uids_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    profiles = np.zeros((len(user_ids), dim), dtype=np.float32)
    valid_profile = np.zeros(len(user_ids), dtype=bool)
    
    for uid, recent_items in recent_by_uid.items():
        if uid not in uids_to_idx:
            continue
        idx = uids_to_idx[uid]
        embeds = []
        for item_id in recent_items:
            if item_id in item_to_idx:
                embeds.append(vectors[item_to_idx[item_id]])
        if embeds:
            profiles[idx] = np.mean(embeds, axis=0)
            valid_profile[idx] = True
    
    # Normalize profiles
    norms = np.linalg.norm(profiles, axis=1, keepdims=True)
    norms[~valid_profile] = 1.0
    profiles = np.where(valid_profile[:, None], profiles / norms, profiles)
    
    print(f"[04_recs] built profiles in {time.monotonic()-t0:.1f}s", flush=True)
    
    # Batch computation of top-k per user
    print("[04_recs] computing similarities (batched)...", flush=True)
    t0 = time.monotonic()
    
    for batch_start in range(0, len(user_ids), batch_size):
        batch_end = min(batch_start + batch_size, len(user_ids))
        batch_uids = user_ids[batch_start:batch_end]
        batch_profiles = profiles[batch_start:batch_end]  # (B, dim)
        batch_valid = valid_profile[batch_start:batch_end]  # (B,)
        
        if not batch_valid.any():
            # All users in this batch are cold - skip
            n_cold += len(batch_uids)
            continue
        
        # Compute similarities: (N, B) = vectors @ batch_profiles.T
        sims = vectors @ batch_profiles.T  # (n_items, batch_size) float32
        
        # Build exclude mask for this batch
        for bi, uid in enumerate(batch_uids):
            # Exclude items user has already played
            played = played_by_uid.get(uid, set())
            for item_id in played:
                if item_id in item_to_idx:
                    sims[item_to_idx[item_id], bi] = -1e9
        
        # Get top-k indices per column
        # argpartition gives top-k unsorted, then sort
        top_idx = np.argpartition(sims, -top_k, axis=0)[-top_k:, :]  # (k, B)
        top_idx = top_idx.T  # (B, k)
        
        # Sort each row by similarity
        for bi in range(len(batch_uids)):
            if not batch_valid[bi]:
                # Cold user - skip (no recommendations)
                n_cold += 1
                continue
            
            sorted_order = np.argsort(-sims[top_idx[bi], bi])
            for rank in range(top_k):
                idx = top_idx[bi, sorted_order[rank]]
                results.append({
                    "user_id": int(batch_uids[bi]),
                    "item_id": int(idx_to_item[idx]),
                    "score": float(sims[idx, bi]),
                    "rank": rank + 1,
                })
    
    print(f"[04_recs] similarity computation done in {time.monotonic()-t0:.1f}s", flush=True)
    return results, n_cold


def write_partitioned_recs(results: list[dict], out_dir: Path, n_partitions: int = 100) -> dict:
    """Write recommendations to partitioned parquet files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by partition
    partitions: dict[int, list[dict]] = {p: [] for p in range(n_partitions)}
    for r in results:
        # Partition key: (user_id // 100) % 100
        p = (r["user_id"] // 100) % n_partitions
        partitions[p].append(r)
    
    # Write each partition
    t0 = time.monotonic()
    for p in range(n_partitions):
        if not partitions[p]:
            continue
        rows = partitions[p]
        table = pa.table({
            "user_id": pa.array([r["user_id"] for r in rows], type=pa.uint32()),
            "item_id": pa.array([r["item_id"] for r in rows], type=pa.uint32()),
            "score": pa.array([r["score"] for r in rows], type=pa.float32()),
            "rank": pa.array([r["rank"] for r in rows], type=pa.int32()),
        })
        out_file = out_dir / f"part-{p:03d}.parquet"
        pq.write_table(table, out_file)
    
    print(f"[04_recs] wrote {len(results):,} rows to {len([p for p in partitions if p])} partitions in {time.monotonic()-t0:.1f}s", flush=True)
    
    # Manifest
    manifest = {
        "n_users": len(set(r["user_id"] for r in results)),
        "n_recs": len(results),
        "n_partitions": len([p for p in partitions if p]),
        "partition_key": "(user_id // 100) % N",
        "n_partitions_requested": n_partitions,
        "total_bytes": sum(f.stat().st_size for f in out_dir.glob("*.parquet")),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    manifest_file = out_dir.parent / "recs_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))
    print(f"[04_recs] manifest -> {manifest_file}", flush=True)
    
    return manifest


def recs(
    force: bool = False,
    num_recent: int = 20,
    min_train_len: int = 20,
    top_k: int = 20,
    n_partitions: int = 100,
    batch_size: int = 256,
) -> dict:
    """Main entry point for recommendation generation."""
    root = Path(os.environ.get("RECOMUSIC_DATA_DIR", "data"))
    raw = root / "raw"
    artifacts = Path(os.environ.get("RECOMUSIC_ARTIFACTS_DIR", str(root / "artifacts")))
    
    listens = raw / "listens.parquet"
    emb_path = artifacts / "track_emb.parquet"
    out_dir = artifacts / "recs"
    
    if not listens.is_file():
        raise FileNotFoundError(f"{listens} not found — run 01_ingest first")
    if not emb_path.is_file():
        raise FileNotFoundError(f"{emb_path} not found — run 02_item2vec first")
    
    manifest_path = out_dir.parent / "recs_manifest.json"
    if manifest_path.is_file() and not force:
        print(f"[04_recs] recommendations already present — use --force to regenerate: {emb_path.parent}")
        return json.loads(manifest_path.read_text())
    
    t0 = time.monotonic()
    
    # Load embeddings
    print(f"[04_recs] loading track embeddings from {emb_path}", flush=True)
    vectors, item_to_idx, idx_to_item = load_track_embeddings(emb_path)
    print(f"[04_recs] loaded {len(idx_to_item):,} item embeddings, dim={vectors.shape[1]}", flush=True)
    
    # Get valid users
    valid_user_ids = get_valid_users(listens, min_train_len)
    
    # Compute recommendations
    results, n_cold = compute_recs_for_users(
        listens, vectors, item_to_idx, idx_to_item,
        valid_user_ids, num_recent, min_train_len, top_k, batch_size
    )
    
    # Write partitioned output
    manifest = write_partitioned_recs(results, out_dir, n_partitions)
    manifest["elapsed_sec"] = round(time.monotonic() - t0, 1)
    manifest["num_recent_used"] = num_recent
    manifest["top_k"] = top_k
    manifest["n_cold_users"] = n_cold
    
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="regenerate even if artifacts exist")
    parser.add_argument("--num-recent", type=int, default=20, help="most recent tracks for profile (default: 20)")
    parser.add_argument("--min-train-len", type=int, default=20, help="min plays required (default: 20)")
    parser.add_argument("--top-k", type=int, default=20, help="recommendations per user (default: 20)")
    parser.add_argument("--partitions", type=int, default=100, help="number of output partitions (default: 100)")
    parser.add_argument("--batch-size", type=int, default=256, help="computation batch size (default: 256)")
    args = parser.parse_args()
    try:
        result = recs(
            force=args.force,
            num_recent=args.num_recent,
            min_train_len=args.min_train_len,
            top_k=args.top_k,
            n_partitions=args.partitions,
            batch_size=args.batch_size,
        )
        print(f"\n[04_recs] complete: {result['n_users']} users, {result['n_recs']} recommendations in {result['elapsed_sec']}s", flush=True)
    except Exception as exc:
        print(f"04_RECS FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())