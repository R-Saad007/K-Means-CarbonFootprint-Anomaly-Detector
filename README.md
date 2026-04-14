# AI-Driven Carbon Efficiency Benchmarking

A production-ready FastAPI microservice that transforms passive telecom network
telemetry into actionable environmental and financial ROI. It calculates
site-level carbon footprints, clusters sites via K-Means to derive dynamic
"digital peer" baselines, and flags anomalous power consumption as structured
tickets for the NOC (Network Operations Center).

The service is designed from day one to be consumed by downstream agentic
workflows and LLM-based NOC assistants — every endpoint returns structured
JSON with stable field names and explicit units.

---

## 1. Architecture at a Glance

```
                ┌──────────────────────────────────────────────────┐
                │                  FastAPI (app.main)              │
                │   /healthz    /api/v1/ingest   /api/v1/cluster   │
                │   /api/v1/anomalies   /api/v1/sites/{id}/peer-…  │
                └─────────────┬────────────────────────────────────┘
                              │  depends_on
                ┌─────────────▼─────────────┐
                │   app.storage.DataStore   │  ← in-memory singleton
                │  • siteinfra  DataFrame   │
                │  • performance DataFrame  │
                │  • ClusterState           │
                │  • anomalies: List[dict]  │
                └─────────────┬─────────────┘
                              │ read / write
     ┌────────────┬───────────┼────────────┬─────────────┐
     │            │           │            │             │
┌────▼────┐  ┌────▼────┐ ┌────▼────┐  ┌────▼────┐   ┌────▼────┐
│Ingestion│  │ Carbon  │ │Cluster- │  │ Anomaly │   │ Peer    │
│ service │→ │ service │→│  ing    │→ │ service │   │Analysis │
└─────────┘  └─────────┘ └─────────┘  └─────────┘   └─────────┘
   CSV ETL    Scope 1+2   K-Means+     20% rule +     per-site
   canonical  per site    silhouette   Hub bypass     context
```

### Source Layout

```
app/
├── main.py              FastAPI app, logging, router wiring
├── config.py            Pydantic-settings; env-var driven tuning
├── schemas.py           Pydantic I/O contracts
├── storage.py           DataStore singleton (thread-safe via RLock)
├── api/
│   └── routes.py        All /api/v1 endpoints
└── services/
    ├── ingestion.py     CSV → canonical DataFrames
    ├── carbon.py        calculate_site_co2() + vectorised variant
    ├── clustering.py    KMeans + silhouette + per-cluster baselines
    └── anomaly.py       Threshold + Hub exception + ticket builder
tests/
├── conftest.py          Seeded in-memory store with digital peers + hub
├── test_carbon.py       CO2 formula correctness
├── test_clustering.py   K-Means grouping & baseline semantics
└── test_anomaly.py      20% threshold + Hub exemption + payload shape
Dockerfile               Non-root uvicorn deployment
requirements.txt
```

---

## 2. Endpoints

All endpoints are versioned under `/api/v1` so that breaking changes can
coexist with the stable surface agents depend on.

| Method | Path                                     | Purpose                                                     |
|--------|------------------------------------------|-------------------------------------------------------------|
| POST   | `/api/v1/ingest`                         | Trigger ETL of the two source sheets.                       |
| POST   | `/api/v1/cluster`                        | Train K-Means, compute per-cluster baselines.               |
| GET    | `/api/v1/anomalies?refresh=true`         | Evaluate & return NOC tickets.                              |
| GET    | `/api/v1/sites/{site_id}/peer-analysis`  | Cluster context for one site (mandatory for agent workflows). |
| GET    | `/healthz`                               | Liveness probe.                                             |
| GET    | `/docs`                                  | Interactive OpenAPI UI.                                     |

### Typical lifecycle

```bash
curl -X POST http://localhost:8000/api/v1/ingest
curl -X POST http://localhost:8000/api/v1/cluster
curl      http://localhost:8000/api/v1/anomalies
curl      http://localhost:8000/api/v1/sites/S104/peer-analysis
```

---

## 3. Data Ingestion — Why Only Two Sheets

Per the client spec, this service **must not** ingest hourly or real-time
telemetry. We are strictly limited to:

1. `AxIn headers_annotated.xlsx - siteinfra.csv` — static inventory (Grid /
   DG / Solar / Hub flag / region / cooling type).
2. `AxIn headers_annotated.xlsx - performancedaily.csv` — daily energy,
   fuel, and load telemetry.

**Why the constraint?** The client's target deployment footprint is small, and
hourly telemetry would balloon both RAM and compute costs during daily
evaluations. Daily aggregation is sufficient to detect sustained thermal,
cooling, or generator anomalies — the ones that dominate carbon and OpEx.

**How we implement it.** `app/services/ingestion.py` canonicalises columns
to a stable internal schema (`site_id`, `is_hub_site`, `grid_kwh`, …) via
fuzzy matching (`_find_col`). Vendor exports historically drift
(e.g., `DG_Capacity_kVA` vs `DieselGenCapacity`); fuzzy matching means a new
vendor feed rarely breaks the pipeline.

**Tradeoff.** Fuzzy matching vs. a strict schema:
- **Chosen**: tolerance — accept many column name variants, fall back to
  sensible defaults (`0.0` for missing capacity; default grid EF for unknown
  region). Better developer / operator ergonomics.
- **Rejected**: strict schema with hard failures. Too brittle given that the
  annotated headers keep evolving across releases.

When fuel consumption is missing but DG energy is known, we derive litres
using the industry-standard **0.27 L/kWh** conversion for telecom gensets at
typical loads. This keeps Scope 1 emissions non-zero rather than silently
dropping them.

---

## 4. Carbon Calculation

`app/services/carbon.py::calculate_site_co2()` implements the reference
formula exactly:

```python
dg_co2_kg    = fuel_consumed_L * 2.68          # Scope 1 — diesel
grid_co2_kg  = grid_kwh * provincial_EF        # Scope 2 — grid
solar_co2_kg = 0                               # Scope 1 offset — solar
total_co2_kg = dg_co2_kg + grid_co2_kg
```

### Provincial emission factors

A conservative set of Canadian provincial grid EFs ships in
`settings.grid_emission_factors` (kg CO2 per kWh). Unknown regions fall back
to `default_grid_ef_kg_per_kwh = 0.40`. These values are overridable per
deployment via `CARBON_GRID_EMISSION_FACTORS` or by editing `config.py`.

**Why the fallback?** Silently returning zero for Scope 2 would hide the
majority of emissions at any grid-heavy site. A visible, conservative default
is safer than an invisible zero.

### Vectorised vs. per-site

Two entry points exist intentionally:
- **`calculate_site_co2(site_id, date, store)`** — matches the spec's
  reference signature; used by per-site lookups and tests.
- **`compute_co2_dataframe(perf_df, infra_df)`** — vectorised form used by
  clustering and anomaly evaluation so we never loop rows in pandas.

---

## 5. K-Means Clustering & Baselining

### Feature set

Features are deliberately restricted to *operational and topological*
characteristics — **never** CO2 itself. Clustering on CO2 would trivially
hide the very anomalies we want to detect.

```
total_load_kwh        daily operational load
grid_capacity_kw      installed grid capacity
dg_capacity_kw        installed genset capacity
solar_capacity_kw     installed solar capacity
has_grid / has_dg / has_solar    boolean presence flags
cooling_type          one-hot encoded (air / freecooling / unknown / …)
```

All numeric features are `StandardScaler`-normalised before K-Means.

### Choosing K

`_pick_k()` selects K via the **silhouette score** over
`[kmeans_k_min=3, kmeans_k_max=8]` by default, or honours an explicit
`CARBON_KMEANS_K` env var.

**Tradeoff.**
- **Chosen**: auto-K via silhouette. Adapts as the fleet grows or as new site
  classes appear, no manual re-tuning needed.
- **Rejected**: hardcoded K. Simpler, but requires operator intervention any
  time the fleet composition shifts.
- **Rejected**: elbow method. More subjective; silhouette provides a single
  comparable number.

Degenerate sample sizes (very small fleets during bootstrapping) fall back
gracefully to `min(samples, 2)` to avoid sklearn errors.

### Baseline = median, not mean

Each cluster's baseline is the **median** `total_co2_kg` of its members.
The mean is also recorded for transparency.

**Why median?** A single egregious anomaly on one site would yank the mean
upward, raising the baseline and hiding the very outlier we want to flag.
The median is robust to outliers and is therefore the correct fixed point
for threshold comparisons.

---

## 6. Anomaly Detection Rules

Implemented in `app/services/anomaly.py::evaluate_anomalies()`:

1. **Threshold check first** — flag sites where `total_co2 > baseline × 1.20`
   (20% configurable via `CARBON_ANOMALY_THRESHOLD_PCT`).
2. **Hub Site Exception** — if `is_hub_site == True`, the alert is
   *suppressed* regardless of variance. Hubs naturally carry higher baseline
   variance due to heavy baseband processing and aggregation traffic; leaving
   them in would generate constant false positives and erode NOC trust.
3. **Baseline guard** — clusters with baseline ≤ 0 are skipped; variance
   against zero is mathematically meaningless.

### Ticket payload

```json
{
  "ticket_id": "uuid",
  "site_id": "S104",
  "evaluation_date": "2026-04-14",
  "cluster_id": 2,
  "total_co2_kg": 612.55,
  "cluster_baseline_co2_kg": 500.00,
  "variance_pct": 22.51,
  "threshold_pct": 20.0,
  "peer_count": 49,
  "diagnosis": "Site S104 is consuming 22.5% more power than its 49 digital peers with identical traffic and weather conditions. Investigate cooling efficiency, DG runtime, and any recent hardware changes on this site.",
  "is_hub_site": false,
  "created_at": "…"
}
```

The `diagnosis` string is deliberately written to read well in a Slack
notification or LLM-summarised NOC feed. Agents can use structured fields for
action, and humans get context without extra parsing.

---

## 7. Storage Model — Why In-Memory

`app/storage.py::DataStore` is an in-process singleton wrapping pandas
DataFrames and the current `ClusterState`. Access is guarded by an `RLock`.

**Tradeoff.**
- **Chosen**: in-memory store. Daily workloads easily fit in RAM (thousands
  of sites × a handful of KB each). Zero external dependencies for POC and
  first-production deployments.
- **Rejected for v1**: Redis / Postgres. Overkill for the daily cadence; we
  would pay latency and ops burden for no functional gain.
- **Future-ready**: the `DataStore` class is the only place any service
  reaches for state. Swapping the backing store to Redis or Postgres requires
  changing one file, not the entire codebase.

**Consequence**: a process restart drops state — mitigated by the fact that
`/api/v1/ingest` + `/api/v1/cluster` are cheap and idempotent, and the
daily cadence makes "re-ingest on boot" a trivial recovery pattern.

---

## 8. API Design Decisions

- **`/api/v1` prefix** — future-proofs breaking changes.
- **POST for ingest & cluster, GET for queries** — follows REST semantics;
  POST actions are non-idempotent-looking to discourage accidental retries.
- **409 CONFLICT** when `/anomalies` is hit before `/cluster` — surfaces a
  workflow error clearly rather than returning a misleading empty list.
- **`refresh=true` default on `/anomalies`** — the NOC always wants the
  latest evaluation; set `refresh=false` for cheap repeat reads of the last
  run, useful for paging UIs.
- **Everything returns Pydantic models** — guarantees type-safety on both
  sides of the wire and auto-generates the OpenAPI schema consumed by agents.

---

## 9. Testing Strategy

`tests/conftest.py` builds a **fully synthetic** `DataStore` — 5 operational
sites partitioned into two digital-peer groups plus one Hub Site. No CSVs
are required to run the suite.

Coverage:

| File                     | What it proves                                                         |
|--------------------------|------------------------------------------------------------------------|
| `test_carbon.py`         | Diesel × 2.68; solar = 0; provincial EF; unknown-region fallback.      |
| `test_clustering.py`     | Digital peers land together; baselines are exactly the cluster median. |
| `test_anomaly.py`        | 20% threshold boundary; Hub exemption is unconditional; ticket shape.  |

```bash
python -m pytest -q     # 16 passed, ~20 s
```

**Why synthetic fixtures instead of sample CSVs?** Tests must be deterministic
and hermetic. A sample CSV would couple CI to the specific vendor export
version; synthetic data lets us prove *semantics* and regenerate edge cases
cheaply.

---

## 10. Running Locally

```bash
# 1. install
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 2. drop the two source CSVs into ./data/ (or set CARBON_DATA_DIR)
#    - AxIn headers_annotated.xlsx - siteinfra.csv
#    - AxIn headers_annotated.xlsx - performancedaily.csv

# 3. boot
uvicorn app.main:app --reload --port 8000
```

Browse `http://localhost:8000/docs` for the interactive OpenAPI UI.

### Docker

```bash
docker build -t carbon-bench .
docker run --rm -p 8000:8000 \
  -v "$PWD/data:/app/data" \
  -e CARBON_ANOMALY_THRESHOLD_PCT=20.0 \
  carbon-bench
```

The image runs as a non-root `appuser`, exposes `:8000`, and ships a
`HEALTHCHECK` that pings `/healthz` every 30 s.

---

## 11. Configuration Reference

All knobs are env-var driven, prefix `CARBON_`:

| Env var                              | Default                                          | Notes                                 |
|--------------------------------------|--------------------------------------------------|---------------------------------------|
| `CARBON_DATA_DIR`                    | `./data`                                         | Where the two CSVs live.              |
| `CARBON_DIESEL_EF_KG_PER_L`          | `2.68`                                           | Spec-mandated; do not change lightly. |
| `CARBON_DEFAULT_GRID_EF_KG_PER_KWH`  | `0.40`                                           | Fallback for unknown regions.         |
| `CARBON_KMEANS_K`                    | *(auto via silhouette)*                          | Pin K if desired.                     |
| `CARBON_KMEANS_K_MIN` / `_MAX`       | `3` / `8`                                        | Silhouette search range.              |
| `CARBON_KMEANS_RANDOM_STATE`         | `42`                                             | Reproducibility seed.                 |
| `CARBON_ANOMALY_THRESHOLD_PCT`       | `20.0`                                           | Spec-mandated strict threshold.       |

---

## 12. Known Limitations & Future Work

- **State is in-memory.** Process restart requires a re-ingest. Swap in Redis
  / Postgres when multi-replica deployments are on the roadmap.
- **Hub detection is a boolean.** The spec treats "Hub Site" as a hard
  bypass. A future iteration could attach cluster-specific tolerances per
  topology role (hub vs access vs relay) instead.
- **No streaming telemetry.** By design; the spec forbids hourly ingestion.
  If real-time needs arrive, the right pattern is an async consumer that
  maintains a rolling daily aggregate and posts to `/api/v1/ingest` once
  a day.
- **Scope 3 is out of scope** (no backhaul embodied carbon, no hardware
  lifecycle emissions). Add when inventory sheet carries the fields.
