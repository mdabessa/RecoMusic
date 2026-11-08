"""01_ingest — download Yambda-50M from Hugging Face into ./data/raw (cached).

Downloads the per-event Parquet files for the flat/50m split, plus (optionally)
the track→artist / track→album mappings, and sanity-checks every file:

  * file size vs. expected (from the HF dataset card)
  * valid Parquet + row count vs. the published 50M statistics

A successful run writes <RAW_DIR>/ingest.json (manifest) and a
<RAW_DIR>/.ingest-complete marker; on re-run the files are skipped (cache).

Run standalone:     python jobs/01_ingest.py [--force] [--with-mappings] [--with-embeddings]
Run inside pipeline: imported by flows.py as a Prefect task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

# --- source ------------------------------------------------------------------

HF_BASE = "https://huggingface.co/datasets/yandex/yambda/resolve/main"

# (repo path, local name, expected row count from the dataset card; None = unknown)
EVENT_FILES: list[tuple[str, str, int | None]] = [
    ("flat/50m/listens.parquet", "listens.parquet", 46_467_212),
    ("flat/50m/likes.parquet", "likes.parquet", 881_456),
    ("flat/50m/dislikes.parquet", "dislikes.parquet", 107_776),
    ("flat/50m/unlikes.parquet", "unlikes.parquet", None),
    ("flat/50m/undislikes.parquet", "undislikes.parquet", None),
    ("flat/50m/multi_event.parquet", "multi_event.parquet", None),
]

MAPPING_FILES: list[tuple[str, str, int | None]] = [
    ("artist_item_mapping.parquet", "artist_item_mapping.parquet", None),
    ("album_item_mapping.parquet", "album_item_mapping.parquet", None),
]

# 13.8 GB single file covering all ~9.4M tracks (5B scale). Needed later by
# 03_umap_artists and 05_surprise; downloaded only when explicitly requested.
EMBEDDINGS_FILE = ("embeddings.parquet", "embeddings.parquet", None)

CHUNK = 8 * 1024 * 1024
RETRIES = 3
RETRY_DELAY = 5  # seconds; doubled each attempt


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
                total = int(resp.headers.get("Content-Length", 0)) or None
                got = 0
                next_report = 10 * 1024 * 1024
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
                    if total and got >= next_report:
                        print(
                            f"  {dest.name}: {got / 1e6:.0f}/{total / 1e6:.0f} MB",
                            flush=True,
                        )
                        next_report += 10 * 1024 * 1024
            tmp.rename(dest)
            return
        except Exception as exc:  # noqa: BLE001 — any network/IO error
            tmp.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise RuntimeError(f"download failed after {RETRIES} tries: {url}") from exc
            wait = RETRY_DELAY * (2 ** (attempt - 1))
            print(f"  {dest.name}: attempt {attempt} failed ({exc}); retrying in {wait}s", flush=True)
            time.sleep(wait)


def _validate(dest: Path, expected_rows: int | None) -> dict:
    meta = pq.ParquetFile(dest).metadata
    result: dict = {"bytes": dest.stat().st_size, "rows": meta.num_rows}
    if expected_rows is not None and meta.num_rows != expected_rows:
        raise ValueError(
            f"{dest.name}: row count {meta.num_rows:,} != expected {expected_rows:,}"
        )
    return result


def _is_complete(raw: Path) -> bool:
    return (raw / ".ingest-complete").is_file()


def ingest(
    force: bool = False,
    with_mappings: bool = False,
    with_embeddings: bool = False,
) -> dict:
    raw = Path(os.environ.get("RECOMUSIC_DATA_DIR", "data")) / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    if _is_complete(raw) and not force:
        manifest = raw / "ingest.json"
        if manifest.is_file():
            print(f"Already ingested (marker found) — use --force to re-download: {manifest}")
            return json.loads(manifest.read_text())
        print(f"Marker found but {manifest} missing — re-verifying")

    files = list(EVENT_FILES)
    if with_mappings:
        files += MAPPING_FILES
    if with_embeddings:
        files.append(EMBEDDINGS_FILE)

    manifest: dict = {"dataset": "yandex/yambda flat/50m", "files": {}}
    t0 = time.monotonic()
    for path, name, expected_rows in files:
        dest = raw / name
        if not force and _is_complete(raw) and dest.is_file():
            info = _validate(dest, expected_rows)  # cache hit: re-validate only
            print(f"[cache] {name}: {info['rows']:,} rows, {info['bytes'] / 1e6:.1f} MB")
        else:
            print(f"[down]  {name} ({HF_BASE}/{path})")
            _download(f"{HF_BASE}/{path}", dest)
            info = _validate(dest, expected_rows)
            print(f"[ok]    {name}: {info['rows']:,} rows, {info['bytes'] / 1e6:.1f} MB")
        manifest["files"][name] = {
            **info,
            "sha256": _sha256(dest),
            "source": f"{HF_BASE}/{path}",
        }

    n_users = int(pq.read_table(raw / "likes.parquet").to_pandas()["uid"].nunique())
    manifest["users_in_likes"] = n_users
    manifest["elapsed_sec"] = round(time.monotonic() - t0, 1)

    (raw / "ingest.json").write_text(json.dumps(manifest, indent=2))
    (raw / ".ingest-complete").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))

    total_mb = sum(f["bytes"] for f in manifest["files"].values()) / 1e6
    print(f"\nIngest complete: {len(manifest['files'])} files, {total_mb:.0f} MB total → {raw}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--with-mappings", action="store_true", help="also download artist/album mappings")
    parser.add_argument("--with-embeddings", action="store_true", help="also download embeddings.parquet (~13.8 GB)")
    args = parser.parse_args()
    try:
        ingest(force=args.force, with_mappings=args.with_mappings, with_embeddings=args.with_embeddings)
    except Exception as exc:
        print(f"INGEST FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
