# RecoMusic

A portfolio-grade music recommendation system built on the **Yambda** dataset
(Yandex Music). Heavy compute runs locally with PySpark; AWS serves only
precomputed static artifacts.

> Status: **scaffolding** — see [Roadmap](#roadmap). The plan of record lives in [plan.md](./plan.md).

---

## Flagship features

1. **Recommend music** — personalized top-20 per user (Item2Vec baseline, ALS fallback)
2. **Serendipity** — "discover a distant track you'll probably like": high relevance *and* high novelty
3. **Artist proximity map** — 2D UMAP projection of artists from audio embeddings

## Architecture

```text
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1 — COMPUTE (local, Docker Compose, $0)                    │
│                                                                  │
│   Yambda-50M (download once, cached)                             │
│         ↓                                                        │
│   PySpark jobs (bitnami/spark, local[4], 4GB)                    │
│   01_ingest → 02_item2vec → 03_umap → 04_recs → 05_surprise      │
│         ↓                                                        │
│   Orchestration: Prefect 3 (local, dashboard :4200)              │
│         ↓                                                        │
│   Artifacts: .parquet + duckdb.db                                │
└──────────────────────────────────────────────────────────────────┘
                         ↓ (aws s3 sync)
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2 — SERVING (AWS free tier)                                │
│                                                                  │
│   S3 (private) ── OAC ──→ CloudFront (single HTTPS URL)          │
│   │                        │                                     │
│   ├── /frontend/          ├── React + Vite + TS                  │
│   ├── /data/*.parquet     ├── DuckDB-WASM (browser SQL)          │
│   └── /data/recs/...      └── Plotly.js (artist map)             │
│                                                                  │
│   Zero backend. DuckDB-WASM uses HTTP range requests on S3.      │
└──────────────────────────────────────────────────────────────────┘
                         ↑
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 0 — INFRA (Terraform + Makefile)                           │
│   make up   → terraform apply + s3 sync + CF invalidate          │
│   make down → terraform destroy (bucket included)                │
│   make cost → AWS Cost Explorer snapshot                         │
└──────────────────────────────────────────────────────────────────┘
```

**Core architectural principle:** heavy compute runs **locally** (Docker, `local[*]`),
AWS is used **only to serve precomputed artifacts**. No backend servers, no Lambda,
no cold starts.

## Stack

| Layer          | Tool                      | Rationale                                |
|----------------|---------------------------|------------------------------------------|
| Batch compute  | PySpark (local)           | Core learning target; runs in 4 GB       |
| Orchestration  | Prefect 3                 | Pythonic, 1 container, free dashboard    |
| Containers     | Docker Compose            | One command to run everything            |
| Artifact store | S3 + Parquet              | DuckDB-WASM reads natively               |
| Browser SQL    | DuckDB-WASM               | HTTP range requests → no backend         |
| Frontend       | React + Vite + TypeScript | Modern, fast HMR                         |
| Charts         | Plotly.js                 | Great scatter; WebGL when needed         |
| IaC            | Terraform                 | Reproducible, destroyable infrastructure |
| CI/CD          | GitHub Actions (OIDC)     | No stored AWS keys                       |
| Automation     | Makefile (root)           | `up / down / deploy / pipeline / cost`   |

Explicitly rejected: Lambda, API Gateway, ECS/Fargate, EMR, Glue, NAT Gateway —
all either overkill or outside the free tier.

## Dataset

**Yambda-50M** (Yandex Music, Parquet, ~2 GB) — reduced version of Yambda-5B.
Ships `user_id`, `item_id`, `artist_id`, listen events, likes/dislikes, **audio
embeddings per track**, and official train/val/test splits.

Chosen because it provides audio embeddings (required for serendipity and the
artist map), is pre-split (no leakage), and is Parquet (fast, columnar).
Migration path: prototype on 50M → final runs on 5B once the pipeline is stable.

## ML approach

- **Baseline recs** — Item2Vec (Word2Vec over per-user listen sequences,
  `dim=64`, `window=10`, `epochs=10`); ALS fallback. Evaluated on the official
  temporal split with `precision@10`, `recall@10`, `NDCG@10`; users with fewer
  than 20 training interactions are excluded from test.
- **Artist map** — artist vector = play-count-weighted mean of its tracks' audio
  embeddings, reduced to 2D with UMAP (`n_neighbors=30`, `min_dist=0.1`,
  `metric=cosine`).
- **Serendipity** — `score = α·relevance + β·novelty + γ·genre_diversity`, where
  candidates come from cosine similarity to the user's audio-embedding profile
  and novelty penalizes artists already present in the user's history. Measured
  with `novelty@k` and `serendipity@k`.

## Data contract (frontend ↔ DuckDB-WASM)

Recs are written partitioned by `user_id % 100` so the browser fetches a single
shard per user:

```ts
await db.query(`
  SELECT * FROM read_parquet(
    'https://<cf-domain>/data/recs/part-${userId % 100}.parquet'
  )
  WHERE user_id = ${userId}
  ORDER BY score DESC
  LIMIT 20
`);
```

The browser issues HTTP range requests and downloads only the relevant shard —
no backend, no cold start, no per-request cost. Latency target: <200 ms.

## Quickstart

Requires Docker, Node.js, Python 3.11+, and (for deploy) Terraform + AWS credentials.

```bash
make pipeline   # download data, train models, write artifacts (local, $0)
make up         # terraform apply + s3 sync + CloudFront invalidation
make cost       # AWS Cost Explorer snapshot
make down       # terraform destroy, including the bucket → back to $0
```

> The Makefile targets above are the intended interface and land in the next
> Phase 0 step (see [Roadmap](#roadmap)). Until then the containers can be driven
> directly:
>
> ```bash
> docker compose up -d prefect    # Prefect server + dashboard on http://localhost:4200
> docker compose exec spark python -c "import pyspark; print(pyspark.__version__)"
> ```

## Local pipeline containers

`docker-compose.yml` defines two services built from `pipeline/Dockerfile`
(Python 3.11, PySpark 3.5, Prefect 3, UMAP, DuckDB):

| Service   | Purpose                                                             |
|-----------|---------------------------------------------------------------------|
| `prefect` | Prefect 3 server — orchestration API and dashboard on `:4200`       |
| `spark`   | Long-lived Spark driver (`local[4]`, 4 GB) — jobs are exec'd into it |

Both mount `./data` at `/data` (dataset + artifacts) and `./pipeline` at `/app`,
so code edits do not require a rebuild. Jobs read `RECOMUSIC_DATA_DIR`,
`RECOMUSIC_ARTIFACTS_DIR`, and `PREFECT_API_URL` from the environment. The
`spark` service waits for the Prefect health endpoint before starting.

## Repository layout

```text
RecoMusic/
├── Makefile
├── docker-compose.yml
├── plan.md
├── README.md
├── data/                       # gitignored: raw dataset (raw/) + artifacts/
├── pipeline/
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── requirements.txt        # pyspark, prefect, umap-learn, duckdb, boto3
│   ├── flows.py                # Prefect @flow orchestrating the jobs
│   └── jobs/
│       ├── 01_ingest.py
│       ├── 02_item2vec.py
│       ├── 03_umap_artists.py
│       ├── 04_recs.py
│       └── 05_surprise.py
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── lib/duckdb.ts
│       ├── components/{ArtistMap,Recos,Surprise}.tsx
│       └── App.tsx
├── infra/terraform/
│   ├── main.tf
│   ├── s3.tf
│   ├── cloudfront.tf
│   ├── iam.tf
│   ├── budgets.tf
│   └── outputs.tf
└── .github/workflows/deploy.yml
```

## Cost

| Scenario                    | Est. monthly cost |
|-----------------------------|-------------------|
| First 12 months (free tier) | ~$0               |
| After free tier             | ~$1–3             |
| `make down`                 | $0 (immediate)    |

Hard ceiling: AWS Budgets alert at $5/month. CloudFront uses `PriceClass_100`
and the stack deliberately contains no NAT, EMR, or Lambda.

## Results

Populated as each milestone lands; numbers are recorded from the official
temporal split.

| Model      | precision@10 | recall@10 | NDCG@10 |
|------------|--------------|-----------|---------|
| Item2Vec   | _pending_    | _pending_ | _pending_ |
| ALS        | _pending_    | _pending_ | _pending_ |

| Serendipity metric | Value     |
|--------------------|-----------|
| `novelty@k`        | _pending_ |
| `serendipity@k`    | _pending_ |

## Roadmap

- [x] Repo, `plan.md`, `README.md` with architecture diagram
- [x] `docker-compose.yml` with `spark` and `prefect` services
- [x] `pipeline/Dockerfile` (Python 3.11 + PySpark 3.5 + Prefect 3)
- [ ] Root `Makefile` stubs (`pipeline`, `deploy`, `up`, `down`, `cost`)
- [ ] Ingest + Item2Vec baseline with recorded metrics
- [ ] Artist UMAP map
- [ ] Serendipity ranking + metrics
- [ ] React frontend on localhost
- [ ] AWS deploy (S3 + CloudFront + OAC + budgets)
- [ ] CI/CD via GitHub Actions with OIDC

Full task breakdown: [plan.md](./plan.md).
