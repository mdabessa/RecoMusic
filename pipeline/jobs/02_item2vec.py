"""02_item2vec — build per-user listen sequences, train Item2Vec, and evaluate model.

Reads data/raw/listens.parquet (from 01_ingest), builds one chronological
sequence of item_id per user, and trains a gensim Word2Vec (Item2Vec)
skip-gram over those sequences:

  dim=64, window=10, epochs=10, min_count=5, sg=1, negative=10

Evaluates the model using a temporal split (train/test on timestamp boundary)
with Precision@10, Recall@10, and NDCG@10 metrics.
Exports data/artifacts/track_emb.parquet and data/artifacts/item2vec_metrics.json.

Run standalone:     python jobs/02_item2vec.py [--epochs 10] [--force]
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
from gensim.models import Word2Vec

LAST_TS = 26_000_000
TEST_SIZE = 86_400  # 24h in 5s units
GAP_SIZE = 1_800    # 30m gap between train and test
TEST_TS = LAST_TS - TEST_SIZE
TRAIN_TS = TEST_TS - GAP_SIZE


def build_sequences(listens: Path) -> list[list[str]]:
    """One ordered sequence of item tokens per user (chronological).

    Uses DuckDB so the group-by + within-group sort over the ~350 MB
    listens file stays a single streaming query (a few seconds, low memory).
    """
    con = duckdb.connect()
    rows = con.execute(
        "SELECT list(item_id ORDER BY timestamp) AS seq "
        "FROM read_parquet(?) GROUP BY uid",
        [str(listens)],
    ).fetchall()
    con.close()
    return [[str(item) for item in seq] for seq, in rows]


def train_item2vec(
    sequences: list[list[str]],
    dim: int = 64,
    window: int = 10,
    epochs: int = 10,
    min_count: int = 5,
    workers: int | None = None,
) -> Word2Vec:
    """Train Word2Vec (skip-gram) over per-user listen sequences."""
    workers = workers or min(16, os.cpu_count() or 1)
    model = Word2Vec(
        vector_size=dim,
        window=window,
        min_count=min_count,
        sg=1,
        negative=10,
        batch_words=50_000,
        workers=workers,
        seed=42,
    )
    model.build_vocab(sequences)
    model.train(sequences, total_examples=model.corpus_count, epochs=epochs)
    return model


def export_track_embeddings(model: Word2Vec, out_path: Path) -> None:
    """Export track embeddings to Parquet table (item_id: uint32, embed: list[float])."""
    vocab_keys = list(model.wv.key_to_index.keys())
    item_ids = [int(k) for k in vocab_keys]
    vectors = model.wv.vectors
    dim = vectors.shape[1]

    item_id_arr = pa.array(item_ids, type=pa.uint32())
    vectors_flat = pa.array(vectors.flatten(), type=pa.float32())
    embed_arr = pa.FixedSizeListArray.from_arrays(vectors_flat, dim)

    table = pa.Table.from_arrays([item_id_arr, embed_arr], names=["item_id", "embed"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)
    print(f"[export] track embeddings → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)


def evaluate_item2vec(
    model: Word2Vec,
    listens: Path,
    k: int = 10,
    num_recent: int = 20,
    min_train_len: int = 20,
) -> dict:
    """Evaluate Item2Vec using a temporal split.

    Metrics computed: Precision@k, Recall@k, NDCG@k over users with >= min_train_len
    training interactions and >= 1 test interaction.
    """
    t0 = time.monotonic()
    vocab_keys = list(model.wv.key_to_index.keys())
    item_str_to_idx = {str(k): i for i, k in enumerate(vocab_keys)}
    vectors = model.wv.vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_norm = vectors / norms

    con = duckdb.connect()
    train_df = con.execute(
        "SELECT uid, list(item_id ORDER BY timestamp) as items "
        "FROM read_parquet(?) "
        "WHERE timestamp < ? AND played_ratio_pct >= 50 "
        "GROUP BY uid "
        "HAVING len(items) >= ?",
        [str(listens), TRAIN_TS, min_train_len],
    ).fetchall()

    test_df = con.execute(
        "SELECT uid, list(DISTINCT item_id) as target_items "
        "FROM read_parquet(?) "
        "WHERE timestamp >= ? AND played_ratio_pct >= 50 "
        "GROUP BY uid",
        [str(listens), TEST_TS],
    ).fetchall()
    con.close()

    train_dict = {uid: items for uid, items in train_df}
    test_dict = {uid: set(items) for uid, items in test_df}
    eval_uids = [u for u in test_dict if u in train_dict and len(test_dict[u]) > 0]

    if not eval_uids:
        print("[eval]   Warning: no evaluation users found matching criteria", flush=True)
        return {"p10": 0.0, "r10": 0.0, "ndcg10": 0.0, "eval_users_sampled": 0}

    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))

    batch_size = 512
    for batch_start in range(0, len(eval_uids), batch_size):
        batch_uids = eval_uids[batch_start : batch_start + batch_size]
        user_vecs = []
        batch_train_sets = []

        for uid in batch_uids:
            train_items = train_dict[uid][-num_recent:]
            indices = [item_str_to_idx[str(it)] for it in train_items if str(it) in item_str_to_idx]
            batch_train_sets.append(set(indices))
            if indices:
                u_vec = vectors_norm[indices].mean(axis=0)
                norm = np.linalg.norm(u_vec)
                if norm > 0:
                    u_vec = u_vec / norm
            else:
                u_vec = np.zeros(vectors.shape[1])
            user_vecs.append(u_vec)

        user_vecs_arr = np.array(user_vecs)
        sims = user_vecs_arr @ vectors_norm.T

        for i, idx_set in enumerate(batch_train_sets):
            if idx_set:
                sims[i, list(idx_set)] = -1e9

        top_indices = np.argpartition(sims, -k, axis=1)[:, -k:]

        for i in range(len(batch_uids)):
            row_sims = sims[i, top_indices[i]]
            sorted_order = np.argsort(-row_sims)
            recs_idx = top_indices[i, sorted_order]

            target = test_dict[batch_uids[i]]
            hits = np.array([1.0 if int(vocab_keys[idx]) in target else 0.0 for idx in recs_idx])

            p_val = float(hits.sum() / k)
            n_pos = len(target)
            r_val = float(hits.sum() / min(n_pos, k))

            dcg = float((hits * discounts[:k]).sum())
            idcg = float(discounts[: min(n_pos, k)].sum())
            ndcg_val = dcg / idcg if idcg > 0 else 0.0

            precisions.append(p_val)
            recalls.append(r_val)
            ndcgs.append(ndcg_val)

    p10 = round(float(np.mean(precisions)), 5)
    r10 = round(float(np.mean(recalls)), 5)
    ndcg10 = round(float(np.mean(ndcgs)), 5)
    eval_time = round(time.monotonic() - t0, 1)

    print(
        f"[eval]   temporal split ({len(eval_uids)} users) evaluated in {eval_time}s: "
        f"p10={p10:.5f}, r10={r10:.5f}, NDCG@10={ndcg10:.5f}",
        flush=True,
    )
    return {
        "p10": p10,
        "r10": r10,
        "ndcg10": ndcg10,
        "eval_users_sampled": len(eval_uids),
    }


def item2vec(
    force: bool = False,
    dim: int = 64,
    window: int = 10,
    epochs: int = 10,
    min_count: int = 5,
    workers: int | None = None,
) -> dict:
    root = Path(os.environ.get("RECOMUSIC_DATA_DIR", "data"))
    raw = root / "raw"
    artifacts = Path(os.environ.get("RECOMUSIC_ARTIFACTS_DIR", str(root / "artifacts")))
    listens = raw / "listens.parquet"
    out_model = artifacts / "item2vec_model.bin"
    out_track_emb = artifacts / "track_emb.parquet"
    out_metrics = artifacts / "item2vec_metrics.json"

    if not listens.is_file():
        raise FileNotFoundError(f"{listens} not found — run 01_ingest first")

    if out_model.is_file() and out_track_emb.is_file() and out_metrics.is_file() and not force:
        print(f"Model and artifacts already present — use --force to retrain: {out_model}")
        return json.loads(out_metrics.read_text())

    t0 = time.monotonic()
    print(f"[seq]   building per-user sequences from {listens}", flush=True)
    sequences = build_sequences(listens)
    n_users = len(sequences)
    n_items = sum(len(s) for s in sequences)
    print(f"[seq]   {n_users:,} users, {n_items:,} listens ({time.monotonic() - t0:.1f}s)", flush=True)

    print(
        f"[train] Word2Vec dim={dim} window={window} epochs={epochs} "
        f"min_count={min_count} sg=1 negative=10",
        flush=True,
    )
    t1 = time.monotonic()
    model = train_item2vec(
        sequences,
        dim=dim,
        window=window,
        epochs=epochs,
        min_count=min_count,
        workers=workers,
    )
    del sequences  # free token lists memory
    n_items_trained = len(model.wv)
    train_time = round(time.monotonic() - t1, 1)
    print(f"[ok]    vocab={n_items_trained:,} items, trained in {train_time}s", flush=True)

    artifacts.mkdir(parents=True, exist_ok=True)
    model.save(str(out_model))
    print(f"[save]  model → {out_model} ({out_model.stat().st_size / 1e6:.1f} MB)", flush=True)

    export_track_embeddings(model, out_track_emb)

    metrics_res = evaluate_item2vec(model, listens, k=10)

    result = {
        "model": "Word2Vec / Item2Vec",
        "hyperparams": {
            "dim": dim,
            "window": window,
            "epochs": epochs,
            "min_count": min_count,
            "sg": 1,
            "negative": 10,
        },
        "vocab_size": n_items_trained,
        "train_time_sec": train_time,
        "metrics": metrics_res,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    out_metrics.write_text(json.dumps(result, indent=2))
    print(f"[save]  metrics → {out_metrics}", flush=True)

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retrain even if artifacts exist")
    parser.add_argument("--dim", type=int, default=64, help="embedding dimension (default: 64)")
    parser.add_argument("--window", type=int, default=10, help="context window (default: 10)")
    parser.add_argument("--epochs", type=int, default=10, help="training epochs (default: 10)")
    parser.add_argument("--min-count", type=int, default=5, help="drop items seen <N times (default: 5)")
    parser.add_argument("--workers", type=int, default=None, help="Word2Vec threads (default: min(16, cpus))")
    args = parser.parse_args()
    try:
        item2vec(
            force=args.force,
            dim=args.dim,
            window=args.window,
            epochs=args.epochs,
            min_count=args.min_count,
            workers=args.workers,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ITEM2VEC FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
