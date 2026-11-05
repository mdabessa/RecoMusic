# RecoMusic — Project Plan

A portfolio-grade music recommendation system built on the Yambda dataset,
combining PySpark batch compute, a full React frontend, and AWS free-tier
serving via Terraform. Three flagship features:

1. **Recommend music** — personalized track recommendations (baseline + implicit)
2. **Serendipity** — "discover a distant track you'll probably like" (out-of-bubble)
3. **Artist proximity map** — 2D UMAP projection of artists from audio embeddings

**Core architectural principle:** heavy compute runs **locally** (Docker, `local[*]`),
AWS is used **only to serve precomputed artifacts**. Zero backend servers,
zero Lambda, zero cold starts.

---

## 1. Goals & Non-Goals

### Goals

- End-to-end pipeline: raw data → trained models → static artifacts → public site
- Reproducible: `make pipeline && make up` from a clean clone
- Cheap to run, cheap to tear down: `make down` returns cost to $0
- Demonstrates: PySpark, ML, UMAP, DuckDB, React/TS, Terraform, CI/CD, IaC
- Honest evaluation (temporal split, novelty, serendipity metrics)

### Non-Goals

- Real-time serving / online learning
- User auth, accounts, or personal data
- Production SLAs, multi-region, HA
- Heavy infra (Kubernetes, EMR, SageMaker endpoints)

---

## 2. Dataset

**Yambda-50M** (Yandex Music, Parquet). Reduced version of Yambda-5B.
Contains: `user_id`, `item_id`, `artist_id`, listen events, likes/dislikes,
**audio embeddings per track**, and official train/val/test splits.

- Size: ~2 GB
- Users: ~1M / Tracks: ~9.4M (in 5B; ~similar scale in 50M sample)
- Why: ships audio embeddings (essential for serendipity + artist map),
  pre-split (no leakage), Parquet (fast).

**Migration path:** prototype on 50M → final runs on 5B once pipeline is stable.

---

## 3. Architecture

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

---

## 4. Stack

| Layer            | Tool                        | Rationale                                    |
|------------------|-----------------------------|----------------------------------------------|
| Batch compute    | PySpark (local)             | Core learning target; runs in 4 GB           |
| Orchestration    | Prefect 3                   | Pythonic, 1 container, free dashboard        |
| Containers       | Docker Compose              | One command to run everything                |
| Artifact store   | S3 + Parquet                | DuckDB-WASM reads natively                   |
| Browser SQL      | DuckDB-WASM                 | HTTP range requests → no backend             |
| Frontend         | React + Vite + TypeScript   | Modern, fast HMR                             |
| Charts           | Plotly.js (+ deck.gl opt.)  | Great scatter; WebGL if needed               |
| IaC              | Terraform                   | As requested                                 |
| CI/CD            | GitHub Actions (OIDC)       | No stored AWS keys                           |
| Automation       | Makefile (root)             | `up / down / deploy / pipeline / cost`       |

**Explicitly rejected:** Lambda, API Gateway, ECS/Fargate, EMR, Glue,
NAT Gateway — all either overkill or outside free tier.

---

## 5. Features & ML Pipeline

### 5.1 Baseline recommendation

- **Model:** Item2Vec (gensim Word2Vec over per-user listen sequences).
  Fallback: `pyspark.ml.recommendation.ALS` (implicit).
- **Evaluation:** official Yambda temporal split. Metrics:
  `precision@10`, `recall@10`, `NDCG@10`.
- **Guardrail:** filter users with <20 training interactions from test.

### 5.2 Artist proximity map

- Compute artist vector = weighted mean of audio embeddings of its tracks
  (weight = play count, and optionally likes as extra weight).
- Reduce to 2D with **UMAP** (`n_neighbors=30`, `min_dist=0.1`, `metric=cosine`).
- Export `artist_umap.parquet` (artist_id, x, y, name, play_count, top_tracks).
- Frontend renders with Plotly scatter; hover shows top tracks.

### 5.3 Serendipity / "out-of-bubble" discovery

Two spaces matter: **preference** (what user engages with) and
**content** (audio similarity). Serendipity = maximize both while fighting
their natural conflict.

1. user_profile = weighted mean of audio embeddings of liked/played tracks
2. candidates = top-N by cosine similarity to user_profile (relevance)
3. novelty filter: drop candidates whose artist already appears in history (or whose genre/cluster differs from history centroid)
4. final score = α·relevance + β·novelty + γ·genre_diversity

**Optional stretch:** "bridge" recommendations — shortest path in the
artist similarity graph from the user's centroid to a distant genre cluster.
Frontend shows the "journey" (intermediate bridge tracks), not just the
destination.

**Metrics:** `novelty@k` (mean distance to history) and
`serendipity@k` (high predicted preference AND high novelty).

---

## 6. Repository Layout

```text
RecoMusic/
├── Makefile
├── docker-compose.yml
├── plan.md
├── README.md
├── pipeline/
│   ├── Dockerfile
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
│       ├── lib/
│       │   └── duckdb.ts
│       ├── components/
│       │   ├── ArtistMap.tsx
│       │   ├── Recos.tsx
│       │   └── Surprise.tsx
│       └── App.tsx
├── infra/terraform/
│   ├── main.tf
│   ├── s3.tf
│   ├── cloudfront.tf
│   ├── iam.tf
│   ├── budgets.tf
│   └── outputs.tf
└── .github/workflows/
    └── deploy.yml
```

---

## 7. Phases & Tasks

### Phase 0 — Scaffolding

- [/] Create repo, `plan.md`, `README.md` with architecture diagram
- [ ] `docker-compose.yml` with `spark` and `prefect` services
- [ ] Root `Makefile` stubs (`pipeline`, `deploy`, `up`, `down`, `cost`)
- [ ] `pipeline/Dockerfile` (Python 3.11 + PySpark 3.5 + Prefect 3)
- [ ] `pip install` everything; smoke test `pyspark` import

### Phase 1 — Ingest + Baseline ML

- [ ] `01_ingest.py`: download Yambda-50M to `./data/raw`, cache, sanity-check row counts
- [ ] `02_item2vec.py`: build per-user sequences → train Word2Vec (dim=64, window=10, epochs=10)
- [ ] Evaluation job: temporal split, `p10/r10/NDCG@10`
- [ ] Persist `track_emb.parquet`, `item2vec_metrics.json`
- [ ] **Milestone: baseline metrics recorded**

### Phase 2 — Artist Map

- [ ] `03_umap_artists.py`: aggregate embeddings per artist, UMAP 2D
- [ ] Export `artist_umap.parquet`
- [ ] Static local plot (matplotlib) as sanity check
- [ ] **Milestone: artist map rendered locally**

### Phase 3 — Serendipity

- [ ] `04_recs.py`: top-20 per user, partition by `user_id % 100`
- [ ] `05_surprise.py`: implement relevance + novelty + diversity scoring
- [ ] Metrics: `novelty@k`, `serendipity@k` on validation
- [ ] Partition parquet outputs by `user_id % N` for DuckDB range requests
- [ ] **Milestone: core ML features complete**

### Phase 4 — Frontend

- [ ] Vite + React + TS scaffold; DuckDB-WASM init
- [ ] `ArtistMap.tsx` (Plotly scatter, hover tooltip)
- [ ] `Recos.tsx` (list from DuckDB query on partition parquet)
- [ ] `Surprise.tsx` ("Take me out of my bubble" button)
- [ ] Run frontend against **local** parquet files (no AWS yet)
- [ ] **Milestone: fully functional site on localhost**

### Phase 5 — AWS Deploy

- [ ] Bootstrap: create state bucket `reco-music-tfstate` (versioned, encrypted)
- [ ] `s3.tf`, `cloudfront.tf` (OAC, PriceClass_100), `budgets.tf`, `outputs.tf`
- [ ] `terraform init && terraform apply` from `infra/terraform/`
- [ ] `make deploy`: build frontend → sync to S3 → invalidate CloudFront
- [ ] Verify public URL, verify DuckDB-WASM range requests in DevTools
- [ ] **Milestone: public URL live on AWS**

### Phase 6 — CI/CD + Polish

- [ ] GitHub Actions: build + `aws s3 sync` + invalidate, using **OIDC** (no long-lived keys)
- [ ] `terraform` step in CI optional (guard behind manual approval)
- [ ] Prefect deployment for full pipeline (scheduled weekly)
- [ ] `README.md`: GIF of the site, architecture, quickstart, results table
- [ ] `make down` dry-run — confirm everything is destroyed, cost returns to $0
- [ ] **Milestone: portfolio-ready**

---

## 8. Terraform (key resources)

- **State:** S3 backend, versioned, encrypted, one-time manual bootstrap
- **Bucket:** `force_destroy = true` (allows clean `destroy`)
- **CloudFront:** OAC to S3, `PriceClass_100`, default cert, `redirect-to-https`
- **IAM:** role for GitHub Actions via OIDC (no access keys in repo)
- **Budgets:** $5/month alert emailed on 80% threshold

Deliberately **not** created: VPC, NAT, Lambda, API Gateway, RDS, ECS.

---

## 9. Frontend ↔ Data Contract

DuckDB-WASM reads parquet directly from CloudFront:

```ts
// fetch top-20 recs for a user
await db.query(`
  SELECT * FROM read_parquet(
    'https://<cf-domain>/data/recs/part-${userId % 100}.parquet'
  )
  WHERE user_id = ${userId}
  ORDER BY score DESC
  LIMIT 20
`);
```

- Browser issues HTTP range requests → downloads only the relevant ~2 MB shard
- No backend, no cold start, no per-request cost
- Latency target: <200 ms end-to-end

---

## 10. Cost Model

| Scenario                    | Est. monthly cost |
| --------------------------- | ----------------- |
| First 12 months (free tier) | ~$0               |
| After free tier             | ~$1–3             |
| `make down`                 | $0 (immediate)    |

Hard ceiling: AWS Budgets alert at $5/month.

---

## 11. Risks & Mitigations

| Risk                                  | Mitigation                                             |
| ------------------------------------- | ------------------------------------------------------ |
| PySpark OOM on laptop                 | Use 50M dataset; `local[4]`, 4 GB, tune shuffle        |
| Popularity bias in recs               | Min-plays filter on candidates; log-popularity debias  |
| Data leakage in evaluation            | Use official Yambda temporal split                     |
| CloudFront misconfig exposes bucket   | Enforce OAC; bucket policy denies public access        |
| Orphan resources after experiment     | `force_destroy` on bucket; `make down` in CI weekly    |
| DuckDB-WASM threads require COOP/COEP | Start single-threaded; add headers later if needed     |
| Free tier expiry surprise bill        | Budgets alert; `price_class_100`; no NAT/EMR/Lambda    |
| Terraform state corruption            | Versioned S3 backend + terraform state backups         |

---

## 12. Definition of Done

- [ ] `git clone && make pipeline && make up` produces a working public URL
- [ ] `make down` destroys all AWS resources; `make cost` shows $0
- [ ] README contains: architecture diagram, screenshot/GIF, eval table, quickstart
- [ ] Baseline metrics + novelty/serendipity metrics documented
- [ ] All three flagship features visible in UI: recs, surprise, artist map
- [ ] CI green on `main`; deploy workflow uses OIDC (no static AWS keys)
- [ ] `terraform plan` runs clean (no drift)

---

## 13. Stretch Goals (post-DoD)

- [ ] Migrate pipeline to Yambda-5B
- [ ] "Bridge path" serendipity (multi-hop in artist similarity graph)
- [ ] Time slider on artist map (decade-of-career drift)
- [ ] Item2Vec vs ALS vs LightFM comparison table
- [ ] Optional small backend (FastAPI + DuckDB) if dynamic queries become needed
- [ ] Blog post / dev.to write-up with lessons learned