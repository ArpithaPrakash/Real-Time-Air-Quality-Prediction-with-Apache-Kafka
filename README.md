# Local Kafka (Confluent) Development Stack (KRaft)

This repository provides a small local Kafka development environment running a single-node KRaft broker (controller + broker) using Confluent Platform images and a minimal Python producer/consumer toolset under `phase_1_streaming_infrastructure`.

The README below documents complete setup and verification steps, architectural decisions, and a detailed preprocessing methodology (with business justification) used to simulate and stream air quality sensor data to Kafka.

## Table of contents

- Setup — quick start and KRaft storage formatting
- Architecture & design decisions
- How to run (produce / consume)
- Preprocessing methodology (detailed) — what we do and why (business justification)
- Troubleshooting guide & diagnostics
- Developer notes and next steps

---

## Setup — quick start

Prerequisites

- Docker & Docker Compose installed on your machine.
- Python 3.10+ (the project was developed with Python 3.11/3.12 in mind).
- Git (optional)

Project layout (relevant files)

- `docker-compose.yml` — single-node Kafka (KRaft) container definition (uses host `./data/kafka` for storage)
- `phase_1_streaming_infrastructure/producer.py` — host-side Python producer (sends JSON to topic `air_quality_data`)
- `phase_1_streaming_infrastructure/consumer.py` — host-side Python consumer
- `phase_1_streaming_infrastructure/preprocess_data.py` — fetches UCI Air Quality dataset, preprocesses it and can simulate/produce messages to Kafka
- `requirements.txt` — Python dependencies

- `phase_1_streaming_infrastructure/create_topic.py` — small idempotent helper to create `air_quality_data` using the AdminClient

Minimal Phase 1 install (fast path)

If you only want to run Phase 1 (producer + consumer) without the heavier ML/deep packages, install a small set of packages:

```bash
python3 -m pip install pandas confluent-kafka pyarrow prometheus_client ucimlrepo
```

This is a lighter alternative to `pip install -r requirements.txt` and avoids large packages like `tensorflow` when you only need Phase 1.

Start the KRaft Kafka stack

1. From the project root:

```bash
cd /Users/arpithaprakash/Project
docker compose up -d
```

2. Verify the container is running and the host port is exposed:

```bash
docker compose ps
# expect kafka container running and host port 29092 published to container listener 9094
```

3. If this is your first time using `./data/kafka` (or you previously had a ZooKeeper-backed Kafka), format the data directory for KRaft using a generated CLUSTER_ID. Only do this if you intend to initialize the storage directory.

```bash
# generate a cluster id (outputs a UUID string)
docker run --rm -v "$PWD/data/kafka:/var/lib/kafka/data" --entrypoint /bin/bash confluentinc/cp-kafka:7.4.1 -lc "kafka-storage random-uuid"

# format the storage (replace <cluster-id> with the generated id)
docker run --rm -v "$PWD/data/kafka:/var/lib/kafka/data" --entrypoint /bin/bash confluentinc/cp-kafka:7.4.1 -lc "kafka-storage format --config /etc/kafka/kraft/server.properties --cluster-id <cluster-id> --ignore-formatted"

# note: the Compose file expects CLUSTER_ID to match the value you used when formatting
```

Bootstrap addresses

- Host clients (Python scripts): `localhost:29092` (this is the OUTSIDE listener mapped from the container)
- Inside-container clients / other containers on the compose network: `kafka:9092`

---

## Architecture & design decisions

High-level architecture

- Single-node KRaft mode: the Kafka broker also acts as the controller using the Raft metadata quorum (KRaft). This avoids ZooKeeper and simplifies local development.
- Host-exposed OUTSIDE listener: a second listener is configured and published to host port `29092` so host Python processes connect to a stable localhost endpoint instead of touching container internals.
- On-disk persistence: `./data/kafka` is mounted into the container so data survives container restarts during local development if desired.
- Lightweight Python clients: `producer.py`, `consumer.py`, `preprocess_data.py` use `confluent-kafka` to simulate real producers/consumers.

Why these choices?

- KRaft for local dev: avoids running ZooKeeper; easier to manage a single process and matches modern Kafka deployments.
- Host-visible port: simplifies running Python scripts and debugging from host without forwarding container shell sessions.
- Confluent images: widely used, well-tested images for local dev with tooling parity to production.

---

## How to run (produce / consume)

From `phase_1_streaming_infrastructure` folder:

Produce (simulate & send to Kafka)

- Produce locally (use preprocessed data, printed only):

```bash
# print simulated messages
python phase_1_streaming_infrastructure/preprocess_data.py --count 5
```

- Produce and send to Kafka (use the dedicated producer script):

```bash
# send 10 messages to topic air_quality_data (default bootstrap localhost:29092)
python phase_1_streaming_infrastructure/producer.py --kafka --count 10

# producer useful flags:
# --bootstrap <host:port>   (default: localhost:29092)
# --rate <msgs/sec>         (throttle messages)
# --batch-size <n>          (send in micro-batches)
# --qc                     (export QC CSV summary and exit)
```

Consume

```bash
# consume 5 messages then exit
python phase_1_streaming_infrastructure/consumer.py --count 5

# run continuously (consumer exposes Prometheus metrics on :8000 and writes micro-batched parquet files to data/processed/)
python phase_1_streaming_infrastructure/consumer.py
```

Kafka CLI (inside container)

````bash
# create the topic inside the container
docker compose exec kafka bash -lc "kafka-topics --create --topic air_quality_data --bootstrap-server kafka:9092 --partitions 3 --replication-factor 1"


# Helper: idempotent topic creation from host

```bash
# create topic from host using the helper script (uses AdminClient)
python phase_1_streaming_infrastructure/create_topic.py --topic air_quality_data --bootstrap localhost:29092
````

# produce from inside container

docker compose exec kafka bash -lc "printf '{\"test\":\"in-container-produce\"}\n' | kafka-console-producer --bootstrap-server kafka:9092 --topic air_quality_data"

# consume from inside container

docker compose exec kafka bash -lc "kafka-console-consumer --bootstrap-server kafka:9092 --topic air_quality_data --from-beginning --max-messages 1"

````

Notes about bootstrap addresses

- Host processes should use `localhost:29092`.
- If you run code inside the kafka container, use `kafka:9092`.

---

## Preprocessing methodology (detailed) — what we do and why (business justification)

Goal: produce realistic, validated air-quality sensor events derived from the UCI Air Quality dataset so downstream consumers/analytics systems can be exercised in development and integration tests.

What the preprocessing does (step-by-step)

1. Data source

   - We fetch the UCI Air Quality dataset via `ucimlrepo.fetch_ucirepo(id=360)` which provides a clean, time-indexed dataset of sensor readings from a small network of sensors.

2. Basic cleanup

   - Replace dataset sentinel values (-200) with NA so we treat them as missing rather than numeric extremes. Rationale: -200 is a domain-specific sentinel for missing; treating it as numeric skews statistics and downstream thresholds.

3. Numeric coercion

   - Convert all measurable fields to numeric types where possible. Non-numeric columns (Date/Time) are dropped for numeric processing. Rationale: numeric coercion ensures downstream aggregations and ML models behave deterministically and avoids runtime type errors during normalization.

4. Missing value handling

   - For numeric columns, fill missing values with the column mean (computed on available readings). Rationale: mean imputation preserves overall distribution for synthetic sampling and keeps the simulated stream free of NaNs which may crash simpler consumers.

5. Normalization (Min-Max)

   - Numeric columns are min-max scaled to [0,1] per-column. Rationale: Many downstream demo applications and models expect normalized inputs. Scaling makes the synthetic data more stable across sensors with different ranges.

6. Domain-aware validation and clipping

   - Sensor readings for key pollutants are clipped to reasonable environmental ranges (example: CO: 0.5–5 mg/m³, NOx: 5–100 ppb, NO2: 0–200 µg/m³, Benzene: 0.5–10 µg/m³). Rationale: Clipping prevents unrealistic outliers from leaking into tests and provides a safety net where dataset artifacts might be present.

7. Sample generation
   - Each simulated event is a random sample (row) from the preprocessed dataset, supplemented with an ISO-8601 timestamp to emulate a real-time reading. Rationale: sampling preserves correlations between sensors and is cheap to compute while still being realistic.

Business justification

- Faster integration testing: Simulated but realistic sensor data enables QA and dev teams to iterate quickly without requiring live sensor hardware.
- Safety & repeatability: Fixed preprocessing steps make test inputs deterministic (given a seed) and avoid flaky downstream behavior caused by NaN/format differences.
- Early model validation: Normalized, clipped, and validated samples permit early validation of ML pipelines and alerting rules without needing production traffic.
- Cost-effective: Using a well-known open dataset reduces costs and licensing concerns while providing domain realism.

Customization points (for production-like scenarios)

- Replace mean-imputation with model-based or forward-fill imputation to better reflect sensor bursts or outages.
- Add temporal sampling logic (sliding window, periodic sampling) to match real sensor cadence.
- Add noise models or synthetic drift to emulate sensor degradation over time.

---

## Troubleshooting guide & diagnostics

Common quick checks

- Confirm broker is running and the host port is available:

```bash
docker compose ps
nc -vz localhost 29092
````

- Confirm your host producer/consumer are using the host bootstrap address:

```bash
# override env
KAFKA_BOOTSTRAP_SERVERS=localhost:29092 python phase_1_streaming_infrastructure/producer.py --count 1
```

Broker logs

- Tail broker logs with timestamps for correlation with client timestamps. Look for Controller and LeaderAndIsr messages when you see NotLeader/NotLeaderOrFollower or metadata related errors:

```bash
docker compose logs --no-color --timestamps --tail 300 kafka
```

KRaft CLUSTER_ID / storage reformat

- If the broker fails to start complaining about missing KRaft format or CLUSTER_ID:
  - Generate a cluster id using `kafka-storage random-uuid` (see Setup section)
  - Format storage with that cluster id
  - Supply the same CLUSTER_ID in the Compose environment

NotLeader / metadata timing issues

- When a topic is auto-created, there is a short window where the broker must assign leaders for partitions. Host client metadata may try to produce before the assignment completes which can yield NotLeader errors.

Mitigation strategies

- Increase client retries and backoff (we added conservative defaults in `preprocess_data.py` and recommend similar in `producer.py`):

  - message.send.max.retries: 3
  - retry.backoff.ms: 500
  - request.timeout.ms: 20000

- After creating a topic via CLI, wait a couple of seconds before producing from a host client. Or explicitly query topic metadata until partitions have leaders.

- Restart the kafka container if the cluster state appears inconsistent (only after preserving or intentionally reinitializing data in `./data/kafka`)

Collecting diagnostics

- When opening an issue, include:
  - Timestamped client logs (producer debug or application logs)
  - The fragment of `docker compose logs kafka` around the time of failure
  - The output of `docker compose ps` and `ls -la ./data/kafka`

---

## Developer notes & next steps

- `phase_1_streaming_infrastructure/preprocess_data.py` now supports `--kafka` to push simulated messages to `air_quality_data` and `--count`.
- `phase_1_streaming_infrastructure/producer.py` and `consumer.py` provide simple examples for producing and consuming.
- Tests and CI: consider adding a tiny integration test that brings up Kafka (or a lightweight test double) and verifies produce/consume.
- Production hardening (not implemented here): TLS/SASL, multi-broker cluster, monitoring, and ACLs.

- Note about pandas resample: some preprocessing code used `resample('H')` which emitted a FutureWarning in recent pandas releases; the code has been updated to `resample('h')` for future-proofing.

---

## Phase 2 — Data intelligence & visualizations (phase_2_data_intelligence)

Phase 2 is a self-contained analytics and visualization pipeline located at `phase_2_data_intelligence`. It operates on the preprocessed UCI Air Quality data (the preprocessing functions are kept I/O-free) and provides exploratory time-series analyses and lightweight anomaly detection. The producer in Phase 2 is responsible for writing messages to Kafka; preprocessing functions only clean/prepare data.

Key components (files)

- `phase_2_data_intelligence/preprocess_data.py` — fetches the UCI Air Quality dataset, performs cleaning (replace sentinels, type coercion, resample to hourly with `resample('h')`, mean imputation) and returns a DataFrame. This module does not write to Kafka or disk.
- `phase_2_data_intelligence/producer.py` — host-side producer that samples rows from the preprocessed DataFrame and publishes JSON messages to the `air_quality_data` topic (default bootstrap: `localhost:29092`). Use this to stream test messages for Phase 2 consumers.
- `phase_2_data_intelligence/consumer.py` — host-side consumer that subscribes to `air_quality_data`, processes incoming JSON payloads, and logs processed records. Note: the consumer's log output was recently changed to emit JSON-safe lines (uses `json.dumps(..., default=str)`) to make log parsing and verification deterministic.
- `phase_2_data_intelligence/visualizations.py` — generates `daily_pattern.png` and `weekly_pattern.png` (diurnal & weekly means).
- `phase_2_data_intelligence/correlations.py` — computes Pearson correlation matrix and p-values, saving `correlation_matrix.csv`, `correlation_matrix_pvalues.csv`, and `correlation_matrix.png`.
- `phase_2_data_intelligence/advanced_analytics.py` — ACF/PACF plots, STL seasonal decomposition (24h period) per pollutant, residual-based (MAD) anomaly detection, and writes `anomalies.csv` plus per-pollutant decomposition/anomaly PNGs. Also writes a brief `phase2_summary.md`.

Files produced by Phase 2 (saved under `phase_2_data_intelligence/`)

- `daily_pattern.png`, `weekly_pattern.png`
- `correlation_matrix.csv`, `correlation_matrix_pvalues.csv`, `correlation_matrix.png`
- `autocorrelation.png`, `partial_autocorrelation.png`
- `<pollutant>_decomposition.png` (one per pollutant) and `<pollutant>_anomalies.png`
- `anomalies.csv`
- `phase2_summary.md`

How to run Phase 2 (Phase-2-only end-to-end)

1. Install dependencies (from project root):

```bash
cd /Users/arpithaprakash/Project
python3 -m pip install -r requirements.txt
```

2. (Optional) If you want to exercise the Kafka produce/consume flow locally, ensure the KRaft Kafka broker from `docker-compose.yml` is running and reachable at `localhost:29092`.

3a. Run the Phase 2 producer to publish a small batch of messages to Kafka (default: 10 messages):

```bash
python phase_2_data_intelligence/producer.py
```

3b. Start the Phase 2 consumer (it will run continuously unless `--count` is provided). The consumer log lines are JSON-safe which helps downstream parsing for verification:

```bash
python phase_2_data_intelligence/consumer.py
# or run in background with nohup / systemd as you prefer
```

4. Run the analytics/visualization scripts (these operate on the preprocessed dataset and saved artifacts; they do not depend on the running consumer):

```bash
python phase_2_data_intelligence/visualizations.py
python phase_2_data_intelligence/correlations.py
python phase_2_data_intelligence/advanced_analytics.py
```

Notes and caveats

- The Phase 2 producer is intentionally responsible for writing to Kafka — preprocessing functions in `preprocess_data.py` are cleaning-only and do not perform I/O. This keeps the data preparation logic testable and side-effect free.
- Consumer logging: consumer lines are now emitted using `json.dumps(payload, default=str)` so timestamps and NaNs are serialized to string-friendly representations; this makes automated parsing (for verification or metrics) deterministic. If you need strict JSON numeric types for NaNs, post-process the logs to normalize `"NaN"` back to `null` or numeric NaN where appropriate.
- Correlation p-values may be extremely small; they are written in scientific notation. If a column is constant after preprocessing the Pearson correlation is undefined and the scripts will leave `NaN` in those cells.

---

## Phase 3 — Predictive analytics (phase_3_predictive_analytics)

Phase 3 contains a compact predictive analytics pipeline (training, evaluation, and inference) that operates on the preprocessed UCI Air Quality dataset.

What Phase 3 implements (high level)

- Training & artifacts

  - `train_models.py` — training entrypoint (CLI options: `--target`, `--horizon`, `--models`). Trains XGBoost and/or ARIMA by default and writes model pickles.
  - Output naming convention (examples):
    - XGBoost: `xgboost_{TARGET}_tplus{H}.pkl` (e.g. `xgboost_CO_tplus1.pkl`)
    - ARIMA: `arima_{TARGET}.pkl` (e.g. `arima_CO.pkl`)
  - A manifest (`train_manifest.json`) lists the produced model files, the feature columns used during training and now contains per-feature summary statistics to aid inference and monitoring.

- Preprocessing / engineered features

  - `fetch_and_preprocess_data()` (in `preprocess_data.py`) now includes additional engineered features that are useful for short-horizon forecasting:
    - EMA (exponential moving averages): `<col>_ema_<w>` (defaults: 3,6,12,24)
    - Short-term trend slopes (linear fit over a rolling window): `<col>_slope_<w>` (defaults: 6,12,24)
    - Lag-interaction features (pairwise products of existing lagged columns): e.g. `CO_lag1__NOx_lag1`
  - These features are configurable (flags and windows) from the `fetch_and_preprocess_data` call so you can enable/disable or change spans for experiments.

- Hyperparameter tuning

  - `tune_and_evaluate.py` performs rolling-origin CV (TimeSeriesSplit) and a grid search. The grid was expanded to include `max_depth`, `learning_rate`, `subsample`, and `colsample_bytree` and now trains moderate-sized ensembles during tuning. The script saves the best model and a JSON `tune_report_{TARGET}_h{H}.json` describing CV and holdout metrics.

- Evaluation & confidence intervals

  - `evaluate.py` trains a simple XGBoost model and compares it to a persistence baseline (previous observation). It now supports time-series-aware block-bootstrap CIs to better respect temporal dependence:
    - Use `--use-block-bootstrap` and `--block-size <N>` (default 24 samples) to enable block-bootstrap when computing CI for metric differences.

- Runtime inference consumer

  - `model_inference.py` — Kafka consumer that loads trained artifacts and performs online inference for incoming `air_quality_data` messages.
  - The consumer was instrumented with structured JSON logging and Prometheus metrics (metrics endpoint) and contains a simple training-stat-based drift check. Use `--dry-run` to verify model load and metrics server startup without consuming messages.

Quick commands

Train models (example):

```bash
cd /Users/arpithaprakash/Project/phase_3_predictive_analytics
python3 train_models.py --target CO --horizon 1 --models xgboost,arima
```

Run expanded tuning (example — the grid is moderate-sized and takes longer than a tiny search):

```bash
python3 tune_and_evaluate.py --target CO --horizon 1 --n_splits 3 --holdout 0.2 --outdir .
```

Evaluate with block-bootstrap CI (example):

```bash
python3 evaluate.py --target CO --horizon 1 --test-size 0.2 --use-block-bootstrap --block-size 24 --bootstrap 500
```

Run inference (dry-run + live consumer):

```bash
# dry-run to confirm models load and metrics server starts
python3 model_inference.py --dry-run

# run the consumer to listen to Kafka and log predictions
python3 model_inference.py
```

Notes and best practices

- Feature alignment: If you change the preprocessing feature set (enable/disable EMAs, slopes, or interactions), re-train models and update `train_manifest.json`. The runtime consumer assumes models are trained with the feature ordering in the manifest.
- Baseline strength: For short horizons (t+1), persistence (previous observation) is a strong baseline. The expanded feature set and tuning improved CV MAE, but on holdout CO t+1 the best XGBoost run remained behind persistence in previous experiments — consider sequence models or richer temporal features when persistence dominates.
- Reproducible stats: `phase_3_predictive_analytics/train_manifest.json` embeds per-feature statistics (mean/std/min/max/non_null_count) computed on the dataset used to train models. To regenerate feature stats locally, run the helper script from the repository root:

```bash
python3 .tmp_compute_feature_stats.py
```

- Drift & production: Persist training summary statistics (feature means/stds) at training time and have `model_inference.py` load that snapshot; this is already recommended and will make the consumer's drift checks deterministic in production.
