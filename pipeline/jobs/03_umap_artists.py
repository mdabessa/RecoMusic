"""03_umap_artists — artist proximity map: weighted-mean audio embeddings per artist, reduced to 2D with UMAP.

Reads (from 01_ingest):
  data/raw/listens.parquet              listen events (weight = play count)
  data/raw/artist_item_mapping.parquet  item_id -> artist_id
  data/raw/embeddings.parquet           128-dim audio embeddings per track (~13 GB, streamed row-group by row-group)

For every artist with >= --min-plays total plays of its tracks in the 50M data:
  artist_vec = sum(plays_i * emb_i) / sum(plays_i)      (weighted mean, then L2-normalized)

Reduced to 2D with UMAP (n_neighbors=30, min_dist=0.1, metric=cosine).

Exports:
  data/artifacts/artist_umap.parquet        artist_id, x, y, play_count, top_tracks
  data/artifacts/artist_map.png             static matplotlib scatter (sanity check)
  data/artifacts/artist_umap_manifest.json  params + counts

Note: the Yambda dataset does not publish artist names, so the `name` column
in the plan is unavailable; `top_tracks` carries the display info the
frontend hover tooltip needs.

Run standalone:     python jobs/03_umap_artists.py [--force] [--min-plays 1]
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

UMAP_DEFAULTS = dict(n_neighbors=30, min_dist=0.1, metric="cosine", seed=42)


def _build_artist_pairs(
    listens: Path,
    mapping: Path,
    min_plays: int,
    top_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, list[int]]]:
    """Join listens x mapping x play counts in DuckDB.

    Returns:
      artist_ids   sorted unique artist ids (int32)
      pair_items   item ids of eligible (artist, track) pairs, sorted by item_id (int32)
      pair_artists compact artist index per pair row (int32)
      pair_weights play count per pair row (float64)
      play_counts  total plays per artist, aligned with artist_ids (int64)
      top_tracks   artist_id -> list of item ids (up to top_n, by play count)
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TEMP TABLE item_plays AS "
        "SELECT item_id, COUNT(*)::DOUBLE AS plays "
        "FROM read_parquet(?) GROUP BY item_id",
        [str(listens)],
    )
    con.execute(
        "CREATE TEMP TABLE artist_items AS "
        "SELECT m.artist_id, m.item_id, ip.plays "
        "FROM read_parquet(?) m JOIN item_plays ip USING (item_id)",
        [str(mapping)],
    )
    con.execute(
        "CREATE TEMP TABLE eligible AS "
        "SELECT artist_id FROM artist_items "
        "GROUP BY artist_id HAVING SUM(plays) >= ?",
        [min_plays],
    )

    artist_ids = np.asarray(
        [r[0] for r in con.execute(
            "SELECT artist_id FROM eligible ORDER BY artist_id"
        ).fetchall()],
        dtype=np.int32,
    )
    if artist_ids.size == 0:
        con.close()
        return (
            artist_ids,
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
            {},
        )

    artist_index = {int(a): i for i, a in enumerate(artist_ids)}

    pair_rows = con.execute(
        "SELECT a.item_id, a.artist_id, a.plays "
        "FROM artist_items a JOIN eligible e USING (artist_id) "
        "ORDER BY a.item_id"
    ).fetchall()
    pair_items = np.array([r[0] for r in pair_rows], dtype=np.int32)
    pair_artists = np.array([artist_index[r[1]] for r in pair_rows], dtype=np.int32)
    pair_weights = np.array([r[2] for r in pair_rows], dtype=np.float64)
    del pair_rows

    play_counts = np.asarray(
        [
            p
            for _, p in con.execute(
                "SELECT artist_id, SUM(plays) FROM artist_items "
                "JOIN eligible USING (artist_id) GROUP BY artist_id ORDER BY artist_id"
            ).fetchall()
        ],
        dtype=np.int64,
    )

    top_tracks: dict[int, list[int]] = {}
    for artist_id, tracks in con.execute(
        "SELECT artist_id, list(item_id ORDER BY plays DESC, item_id ASC)[1:?"
        " ] FROM artist_items JOIN eligible USING (artist_id) "
        "GROUP BY artist_id",
        [top_n],
    ).fetchall():
        top_tracks[int(artist_id)] = [int(t) for t in tracks]

    con.close()
    return artist_ids, pair_items, pair_artists, pair_weights, play_counts, top_tracks


def _accumulate_artist_vectors(
    embeddings: Path,
    pair_items: np.ndarray,
    pair_artists: np.ndarray,
    pair_weights: np.ndarray,
    n_artists: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream embeddings.parquet and accumulate weighted sums per artist.

    Returns (acc, wsum): acc is (n_artists, dim) float32 weighted embedding
    sum, wsum is (n_artists,) float64 weight sum. Only items present in the
    pair table contribute.
    """
    pf = pq.ParquetFile(embeddings)
    dim: int | None = None
    acc: np.ndarray | None = None
    wsum = np.zeros(n_artists, dtype=np.float64)

    for rg in range(pf.num_row_groups):
        t = pf.read_row_group(rg, columns=["item_id", "embed"])
        eids = t.column("item_id").to_numpy()

        emb_col = t.column("embed").combine_chunks()
        vals = emb_col.values.to_numpy(zero_copy_only=False)
        offsets = emb_col.offsets.to_numpy()
        if dim is None:
            dim = int(offsets[1] - offsets[0])
            acc = np.zeros((n_artists, dim), dtype=np.float32)
            # sanity: fixed-width list column (all rows same dim)
            if not np.all(np.diff(offsets) == dim):
                raise ValueError("embeddings column has variable width")
        e = vals.reshape(-1, dim).astype(np.float32, copy=False)
        del vals, t

        lo = np.searchsorted(pair_items, eids)
        lo_c = np.minimum(lo, pair_items.size - 1) if pair_items.size else lo
        valid = (lo < pair_items.size) & (pair_items[lo_c] == eids)
        if valid.any():
            idx = lo[valid]
            np.add.at(acc, pair_artists[idx], pair_weights[idx, None] * e[valid])
            np.add.at(wsum, pair_artists[idx], pair_weights[idx])
        if (rg + 1) % 10 == 0 or rg == pf.num_row_groups - 1:
            print(f"[emb]   row group {rg + 1}/{pf.num_row_groups} scanned", flush=True)

    assert acc is not None and dim is not None
    return acc, wsum


def _plot_artist_map(
    x: np.ndarray,
    y: np.ndarray,
    play_counts: np.ndarray,
    n_artists: int,
    out_png: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
    sc = ax.scatter(
        x,
        y,
        s=4,
        c=np.log10(1.0 + play_counts),
        cmap="viridis",
        alpha=0.6,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("log10(plays + 1)")
    ax.set_title(f"Yambda-50M artist proximity map — {n_artists:,} artists (UMAP, cosine)")
    ax.set_xlabel("umap-1")
    ax.set_ylabel("umap-2")
    ax.set_facecolor("white")
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]  sanity-check scatter → {out_png}", flush=True)


def umap_artists(
    force: bool = False,
    min_plays: int = 1,
    n_neighbors: int = UMAP_DEFAULTS["n_neighbors"],
    min_dist: float = UMAP_DEFAULTS["min_dist"],
    top_n: int = 5,
    seed: int = UMAP_DEFAULTS["seed"],
) -> dict:
    root = Path(os.environ.get("RECOMUSIC_DATA_DIR", "data"))
    raw = root / "raw"
    artifacts = Path(os.environ.get("RECOMUSIC_ARTIFACTS_DIR", str(root / "artifacts")))
    listens = raw / "listens.parquet"
    mapping = raw / "artist_item_mapping.parquet"
    embeddings = raw / "embeddings.parquet"
    out_parquet = artifacts / "artist_umap.parquet"
    out_png = artifacts / "artist_map.png"
    out_manifest = artifacts / "artist_umap_manifest.json"

    for required in (listens, mapping, embeddings):
        if not required.is_file():
            raise FileNotFoundError(f"{required} not found — run 01_ingest --with-embeddings first")

    if out_parquet.is_file() and out_png.is_file() and out_manifest.is_file() and not force:
        print(f"Artist map already present — use --force to recompute: {out_parquet}")
        return json.loads(out_manifest.read_text())

    t0 = time.monotonic()
    print(
        f"[pairs] building (artist, track, plays) pairs (min_plays={min_plays}, top_n={top_n})",
        flush=True,
    )
    (
        artist_ids,
        pair_items,
        pair_artists,
        pair_weights,
        play_counts,
        top_tracks,
    ) = _build_artist_pairs(listens, mapping, min_plays, top_n)
    n_artists = artist_ids.size
    print(
        f"[pairs] {n_artists:,} artists, {pair_items.size:,} (artist, track) pairs "
        f"({time.monotonic() - t0:.1f}s)",
        flush=True,
    )
    if n_artists == 0:
        raise RuntimeError("no artists matched min_plays filter")

    print("[emb]   streaming embeddings.parquet (item_id + embed only)", flush=True)
    t1 = time.monotonic()
    acc, wsum = _accumulate_artist_vectors(
        embeddings, pair_items, pair_artists, pair_weights, n_artists
    )
    mapped = int((wsum > 0).sum())
    if mapped < n_artists:
        keep = wsum > 0
        artist_ids = artist_ids[keep]
        play_counts = play_counts[keep]
        acc = acc[keep]
        wsum = wsum[keep]
        n_artists = artist_ids.size
        print(f"[emb]   {mapped:,} artists had at least one embedded track ({time.monotonic() - t1:.1f}s)")
    else:
        print(f"[emb]   {mapped:,} artists had at least one embedded track ({time.monotonic() - t1:.1f}s)")

    # weighted mean, then L2-normalize (UMAP metric is cosine)
    acc = (acc / wsum[:, None]).astype(np.float32)
    norms = np.linalg.norm(acc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = (acc / norms).astype(np.float32)
    del acc

    print(
        f"[umap]  UMAP n_neighbors={n_neighbors} min_dist={min_dist} "
        f"metric=cosine seed={seed} on {X.shape[0]:,} x {X.shape[1]}",
        flush=True,
    )
    t2 = time.monotonic()
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        n_components=2,
        random_state=seed,
        n_jobs=os.cpu_count(),
    )
    coords = reducer.fit_transform(X)
    del X, reducer
    print(f"[umap]  done in {time.monotonic() - t2:.1f}s", flush=True)

    # sort by play_count desc (popular artists first — handy for the frontend)
    order = np.argsort(-play_counts)
    artist_ids = artist_ids[order]
    play_counts = play_counts[order]
    x = coords[order, 0].astype(np.float32)
    y = coords[order, 1].astype(np.float32)

    artifacts.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "artist_id": pa.array(artist_ids, type=pa.int32()),
            "x": pa.array(x, type=pa.float32()),
            "y": pa.array(y, type=pa.float32()),
            "play_count": pa.array(play_counts, type=pa.int64()),
            "top_tracks": pa.array(
                [top_tracks.get(int(a), []) for a in artist_ids],
                type=pa.list_(pa.int32()),
            ),
        }
    )
    pq.write_table(table, out_parquet)
    print(f"[export] artist map → {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)", flush=True)

    _plot_artist_map(x, y, play_counts, n_artists, out_png)

    manifest = {
        "n_artists": n_artists,
        "n_tracks_mapped": int((wsum[order] > 0).sum()),
        "min_plays": min_plays,
        "top_n_tracks": top_n,
        "umap_params": {
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "metric": "cosine",
            "seed": seed,
        },
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    out_manifest.write_text(json.dumps(manifest, indent=2))
    print(f"[save]  manifest → {out_manifest}", flush=True)

    print(f"\nArtist map complete: {n_artists:,} artists in {time.monotonic() - t0:.1f}s → {artifacts}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="recompute even if artifacts exist")
    parser.add_argument("--min-plays", type=int, default=1, help="min total plays per artist (default: 1)")
    parser.add_argument("--n-neighbors", type=int, default=UMAP_DEFAULTS["n_neighbors"], help="UMAP n_neighbors (default: 30)")
    parser.add_argument("--min-dist", type=float, default=UMAP_DEFAULTS["min_dist"], help="UMAP min_dist (default: 0.1)")
    parser.add_argument("--top-n", type=int, default=5, help="top tracks per artist to export (default: 5)")
    parser.add_argument("--seed", type=int, default=UMAP_DEFAULTS["seed"], help="UMAP random state (default: 42)")
    args = parser.parse_args()
    try:
        umap_artists(
            force=args.force,
            min_plays=args.min_plays,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            top_n=args.top_n,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"UMAP_ARTISTS FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
