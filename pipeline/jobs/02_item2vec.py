"""02_item2vec — build per-user listen sequences and train Item2Vec.

Reads data/raw/listens.parquet (from 01_ingest), builds one chronological
sequence of item_id per user, and trains a gensim Word2Vec (Item2Vec)
skip-gram over those sequences:

  dim=64, window=10, epochs=10, min_count=5, sg=1, negative=10

The trained model is saved to data/artifacts/item2vec_model.bin for the
downstream jobs (03_umap_artists, 04_recs, 05_surprise).

Run standalone:     python jobs/02_item2vec.py [--epochs 10] [--force]
Run inside pipeline: imported by flows.py as a Prefect task.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import duckdb
from gensim.models import Word2Vec


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
    out = artifacts / "item2vec_model.bin"

    if not listens.is_file():
        raise FileNotFoundError(f"{listens} not found — run 01_ingest first")

    if out.is_file() and not force:
        print(f"Model already trained — use --force to retrain: {out}")
        return {"model": str(out), "cached": True}

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
    model = train_item2vec(sequences, dim=dim, window=window, epochs=epochs,
                           min_count=min_count, workers=workers)
    del sequences  # free the ~3 GB of token lists before saving
    n_items_trained = len(model.wv)
    print(f"[ok]    vocab={n_items_trained:,} items, trained in {time.monotonic() - t1:.1f}s", flush=True)

    artifacts.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"[save]  model → {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)

    return {
        "model": str(out),
        "cached": False,
        "dim": dim,
        "window": window,
        "epochs": epochs,
        "min_count": min_count,
        "n_users": n_users,
        "n_listens": n_items,
        "vocab_size": n_items_trained,
        "train_time_sec": round(time.monotonic() - t1, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="retrain even if the model exists")
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
