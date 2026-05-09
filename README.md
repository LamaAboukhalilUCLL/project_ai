# AI-Powered PostgreSQL Slow Query Monitor

**Course:** Advanced AI — University Project  
**Authors:** Lama Abou Khalil & Lina  
**Dataset:** Stats Stack Exchange dump (stats.stackexchange.com)

---

## Overview

This project implements a live PostgreSQL slow-query monitor that automatically detects, analyzes, and rewrites slow SQL queries using a multi-component AI pipeline. The system combines a custom-trained neural classifier, a fine-tuned large language model, embedding-based memory, and a real-time Flask dashboard.

The pipeline runs continuously against a live PostgreSQL database. When a slow query is detected, it is passed through four AI components in sequence: a neural classifier scores it, an LLM rewrites it, the classifier re-ranks the rewrites, and the best candidate is verified against the database. Only genuine improvements are stored.

---

## System Architecture

The pipeline has six stages:

```
pg_stat_statements (every 5s)
        ↓
[1] Neural classifier — predicts p_slow + execution time (20 features)
        ↓  (if p_slow ≥ 0.5)
[2] Embedding lookup — check for semantically similar past fix
        ↓
[3] Fine-tuned LLM (sqlcoder-7b QLoRA) — generate 3 candidate rewrites
        ↓
[4] Neural ranker — same classifier scores each candidate, ranked by predicted speedup
        ↓
[5] DB verification — cold + warm timing of top candidate against PostgreSQL
        ↓
[6] Storage — genuine wins stored in embedding history + query_log.json
```

---

## Components

### `classifier/classifier.py`
Trains a multi-task PyTorch neural network on `training_data.csv`.

**Architecture:** shared trunk (Linear 20→64→32, ReLU, Dropout 0.3) feeding two heads:
- Classification head → P(slow), trained with binary cross-entropy
- Regression head → predicted log-execution-time in ms, trained with quantile loss (τ=0.7)

**Joint loss:** `α × BCE + (1 − α) × QuantileLoss`, with α=0.6. Quantile loss was chosen over MSE to reduce systematic under-prediction of slow queries — the asymmetric penalty penalises under-prediction more than over-prediction.

**Input features (20 total):**

| Feature | Type | Description |
|---|---|---|
| `has_select_star` | text | Query uses `SELECT *` |
| `has_like_wildcard` | text | Query has `LIKE '%...%'` |
| `join_count` | text | Number of JOINs |
| `has_subquery` | text | Nested SELECT present |
| `has_group_by` | text | GROUP BY clause present |
| `has_order_by_no_limit` | text | ORDER BY without LIMIT |
| `has_or` | text | OR conditions present |
| `has_function_in_where` | text | Function call in WHERE (prevents index use) |
| `plan_total_cost` | plan | PostgreSQL estimated total cost |
| `plan_rows` | plan | Estimated row count |
| `actual_rows` | plan | Actual rows from EXPLAIN ANALYZE |
| `plan_depth` | plan | Depth of the query plan tree |
| `shared_hit` | plan | Buffer cache hits |
| `shared_read` | plan | Buffer cache misses (disk reads) |
| `has_seq_scan` | plan | Sequential scan present |
| `has_index_scan` | plan | Index scan present |
| `has_bitmap_scan` | plan | Bitmap scan present |
| `has_hash_join` | plan | Hash join present |
| `has_nested_loop` | plan | Nested loop join present |
| `has_merge_join` | plan | Merge join present |

**Training results:** 816 rows, 150 epochs, test accuracy 97.6%, slow recall 1.000, log-MAE 0.716, R²=0.645.

### `classifier/inference.py`
Inference wrapper. Provides `QueryAnalyzer` with two methods:
- `analyze(query, conn)` — returns `{p_slow, predicted_ms, features}`
- `rank_candidates(original, candidates, conn)` — scores each rewrite candidate and returns them sorted by predicted speedup

Plan features require a live DB connection. Falls back to text-only features (plan features zeroed) if the connection is unavailable.

### `classifier/query_classifier.pth`
Saved PyTorch model weights.

### `classifier/scaler.pkl`
Fitted `StandardScaler` for the 20 input features. Must be used at inference time to match training-time normalisation.

### `classifier/model_meta.json`
Training metadata: feature column names, hyperparameters, and per-epoch training history. Used by `inference.py` to reconstruct the model with the correct architecture.

---

### `finetune/finetune_sqlcoder.py`
QLoRA fine-tuning script for `defog/sqlcoder-7b` on SQL optimization pairs.

**Approach:** second-stage domain adaptation. `defog/sqlcoder-7b` was already pre-trained by Defog.ai on 20,000+ human-curated SQL queries across 10 schemas, giving it deep SQL semantics. This script adds a domain-specific adapter layer that teaches the model our specific schema (`posts`, `users`, `votes`) and optimization patterns.

**Method:** 4-bit quantization (NF4, double quant) via BitsAndBytes + LoRA adapter targeting attention projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`). Only 13.6M of 7.25B parameters are trained (0.19%), keeping the base model's SQL knowledge intact while adapting it to our context.

**Hyperparameters:** 3 epochs, batch size 2, gradient accumulation 4 (effective batch 8), learning rate 2e-4, cosine schedule with 5% warmup, fp16.

**Training results:**

| Epoch | Eval loss |
|---|---|
| 1 | 0.0973 |
| 2 | 0.0841 |
| 3 | 0.0873 |

Best eval loss: 0.0841. Training time: ~10 minutes on RTX 5070 Laptop GPU (CUDA 12.9, sm_120 Blackwell).

**Requires:** Python 3.12, `venv312` (see Setup).

### `finetune/llm_optimizer.py`
Inference wrapper for the fine-tuned model. Loads the LoRA adapter on top of the quantized base model and exposes two methods:
- `suggest(query)` — single optimized rewrite (beam search, num_beams=4)
- `suggest_candidates(query, n=3)` — three diverse candidates: one greedy beam, two temperature samples (T=0.7, T=1.0)

Post-processing strips `OFFSET 0` repetition artifacts and validates that output is a syntactically plausible SELECT statement. Falls back to GPT-4o-mini automatically if the fine-tuned model weights are not present.

### `finetune/finetune_pairs.json`
250 slow→optimized SQL query pairs used for fine-tuning. Pairs were generated via GPT-4o with the database schema as context, covering: multi-table JOINs, correlated subqueries, `SELECT *`, `LIKE` wildcards, `ORDER BY` without `LIMIT`, `UNION`, `CASE WHEN`, `EXISTS`/`NOT EXISTS`, CTEs, window functions, and date/time aggregations.

### `finetune/finetune_results_sqlcoder.json`
Training run metadata and per-epoch loss history for the sqlcoder fine-tune.

### `finetune/older_bad_approach/`
Archived earlier attempts using `t5-small` (60M) and `codet5-small`. Both were abandoned due to tokenizer corruption on SQL identifiers and insufficient model capacity. Kept for documentation of the methodology evolution.

### `finetune/build_dataset.py`
Script that calls GPT-4o-mini to generate optimized versions of slow queries measured in `training_data.csv`, producing `finetune_pairs.json`.

---

### `monitor/monitor.py`
The main pipeline. Polls `pg_stat_statements` every 5 seconds and runs the full six-stage pipeline on each new slow query. Writes structured JSON log entries to `dashboard/query_log.json` containing: the original query, classifier output, all ranked candidates with predicted speedups, verified execution time, and measured speedup.

---

### `embeddings/embeddings.py`
Semantic memory using `sentence-transformers/all-MiniLM-L6-v2`. When a fix is verified as genuinely faster, it is stored as a (slow_query, fix, speedup_ms, embedding) tuple. Before generating new candidates, the monitor checks whether a semantically similar query has been fixed before (cosine similarity threshold 0.8) and provides it as context to the LLM.

### `embeddings/history.json`
Persistent store of verified fixes. Each entry contains the slow query, the fix, the measured speedup, and the sentence embedding of the slow query.

---

### `dashboard/app.py`
Flask application serving the monitoring dashboard. Exposes:
- `GET /` — dashboard HTML
- `GET /api/queries` — all log entries normalized to a consistent format (handles both old GPT-pipeline format and new sqlcoder format)
- `GET /api/stats` — aggregate statistics: total monitored, flagged, fixes verified, average speedup, total milliseconds saved

### `dashboard/templates/index.html`
Dashboard front-end. Auto-refreshes every 5 seconds. Shows:
- Five metric cards (total, flagged, fixed, average speedup, total time saved)
- Per-query cards with inline before/after horizontal bar charts showing execution time visually
- Grouped bar chart at the bottom comparing slow vs. verified-fixed execution times per query on a log scale

### `dashboard/query_log.json`
Append-only JSON log of all pipeline events. Each `slow_query_handled` entry records the full pipeline trace for one query.

---

### `queries_generation/all_queries.py`
Executes ~460 hand-crafted SQL queries against the loaded database and records cold-cache time, warm-cache median/std, raw run times, and EXPLAIN ANALYZE plan features for each. Writes `training_data.csv`. Adaptive warm-run count: 5 runs for queries under 500ms, 2 runs for 500ms–2s, 1 run for >2s. Queries exceeding the 10s statement timeout are recorded as 10,000ms with a `timed_out_runs` flag.

### `queries_generation/generate_queries.py`
Generates additional training queries beyond the hand-crafted set. Produces diverse parameterized queries covering 12 pattern categories (CASE WHEN, EXISTS, CTEs, date/time aggregations, UNION, window functions, multi-column GROUP BY, self-joins, etc.) with random parameter sampling so each run produces new queries. Deduplicates against existing `training_data.csv` before measuring.

---

### `training_data.csv`
1,438 rows. Each row is one measured query with 20 features and a `slow`/`fast` label. Label distribution: approximately 60% slow, 40% fast.

### `schema.sql`
PostgreSQL schema for the three tables: `posts`, `users`, `votes`. Includes indexes on primary keys and foreign key columns.

### `loading_data/local_data.py`
Parses Stack Exchange XML dumps (`Users.xml`, `Posts.xml`, `Votes.xml`) and bulk-inserts into PostgreSQL. The loaded database contains approximately 345k users, 425k posts, and 1.7M votes.

### `evaluation_pipeline.py`
End-to-end evaluation comparing three configurations: GPT-only, GPT+verify, and the full pipeline (classifier + LLM + ranker + verify). Results saved to `evaluation_pipeline.csv`.

### `evaluate_predictions.py`
Evaluates classifier predictions against measured ground-truth labels. Computes accuracy, precision, recall, F1, and confusion matrix.

---

## Repository Structure

```
project_ai/
├── classifier/
│   ├── classifier.py          # Model training (PyTorch, 20 features, 2 heads)
│   ├── inference.py           # QueryAnalyzer — analyze + rank_candidates
│   ├── classifier.ipynb       # Training notebook with plots
│   ├── query_classifier.pth   # Saved model weights
│   ├── scaler.pkl             # Fitted StandardScaler
│   └── model_meta.json        # Feature list + training history
│
├── finetune/
│   ├── finetune_sqlcoder.py   # QLoRA fine-tuning of defog/sqlcoder-7b
│   ├── llm_optimizer.py       # Inference wrapper (beam + temperature sampling)
│   ├── finetune_pairs.json    # 250 slow→optimized SQL pairs
│   ├── finetune_results_sqlcoder.json  # Training run results
│   ├── build_dataset.py       # GPT-4o pair generation script
│   ├── build_dataset_with_schema.py   # Schema-aware variant
│   ├── finetune_evalutation.ipynb     # Fine-tune evaluation notebook
│   └── older_bad_approach/    # Archived t5-small / codet5-small attempts
│
├── monitor/
│   └── monitor.py             # Main 6-stage pipeline
│
├── dashboard/
│   ├── app.py                 # Flask API + format normalization
│   ├── query_log.json         # Append-only pipeline event log
│   └── templates/
│       └── index.html         # Live dashboard with charts
│
├── embeddings/
│   ├── embeddings.py          # Sentence-transformer similarity store
│   └── history.json           # Verified fixes with embeddings
│
├── queries_generation/
│   ├── all_queries.py         # Hand-crafted query measurement
│   └── generate_queries.py    # Parameterized template generator
│
├── loading_data/
│   └── local_data.py          # XML → PostgreSQL loader
│
├── evaluation_pipeline.py     # End-to-end pipeline comparison
├── evaluate_predictions.py    # Classifier evaluation script
├── training_data.csv          # 1,438-row labeled dataset
├── schema.sql                 # Database schema
├── requirements.txt           # Python 3.13 venv dependencies
├── requirements_312.txt       # Python 3.12 + CUDA venv dependencies
└── README.md
```

> **Note:** `finetune/sqlcoder-finetuned/` (the fine-tuned model weights, ~800MB) is excluded from version control via `.gitignore`. Re-generate by running `finetune/finetune_sqlcoder.py` with `venv312` active.

---

## Setup

### Prerequisites

- Python 3.13 (project runtime) and Python 3.12 (model training + LLM inference)
- PostgreSQL 16+ (developed against PostgreSQL 18)
- NVIDIA GPU with CUDA 12.8+ for LLM inference (recommended; CPU fallback is very slow)
- ~20 GB free disk space for the sqlcoder-7b model weights

### 1. Create environments

**Project runtime (Python 3.13):**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Model training and LLM inference (Python 3.12 + CUDA):**
```powershell
py -3.12 -m venv venv312
venv312\Scripts\activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
pip install --upgrade transformers peft bitsandbytes accelerate datasets
pip install psycopg2-binary python-dotenv openai sentence-transformers flask pandas scikit-learn
```

> `venv312` is required for `finetune/finetune_sqlcoder.py` and `monitor/monitor.py`. The CUDA nightly build is needed for RTX 40/50 series GPUs (sm_89+, sm_120).

### 2. Set up the database

```powershell
psql -U postgres -h localhost -c "CREATE DATABASE stackexchange_db;"
psql -U postgres -h localhost -d stackexchange_db -f schema.sql
```

### 3. Load the Stack Exchange data

Download `stats.stackexchange.com.7z` from the [Stack Exchange Data Dump](https://archive.org/details/stackexchange). Extract `Users.xml`, `Posts.xml`, `Votes.xml` to the project root. Then:

```powershell
python loading_data/local_data.py
```

Verify row counts:
```powershell
psql -U postgres -d stackexchange_db -c "SELECT (SELECT COUNT(*) FROM users) AS users, (SELECT COUNT(*) FROM posts) AS posts, (SELECT COUNT(*) FROM votes) AS votes;"
```

Expected: ~345k users, ~425k posts, ~1.7M votes.

### 4. Generate training data

```powershell
python queries_generation/all_queries.py
```

Runtime: ~35 minutes. Output: `training_data.csv` (~1,438 rows).

### 5. Train the classifier

Open and run `classifier/classifier.ipynb`, or:
```powershell
python classifier/classifier.py
```

Output: `classifier/query_classifier.pth`, `classifier/scaler.pkl`, `classifier/model_meta.json`.

### 6. Fine-tune the LLM

```powershell
venv312\Scripts\activate
python finetune/finetune_sqlcoder.py
```

Downloads `defog/sqlcoder-7b` (~14.5 GB) on first run, then trains the QLoRA adapter for approximately 10 minutes. Output: `finetune/sqlcoder-finetuned/`.

### 7. Run the monitor

```powershell
venv312\Scripts\activate
python monitor/monitor.py
```

In a separate terminal:
```powershell
venv312\Scripts\activate
python dashboard/app.py
```

Open `http://localhost:5000` for the live dashboard.

---

## Environment Variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

Required only if the fine-tuned model is not present and the system falls back to GPT-4o-mini.

---

## Key Design Decisions

**Why quantile loss instead of MSE for the regression head?**  
Standard MSE regression on execution time systematically under-predicts slow queries because the loss is symmetric — over- and under-prediction are penalised equally. With τ=0.7, the quantile loss penalises under-prediction more, biasing predictions upward and reducing false negatives for the slow class.

**Why second-stage fine-tuning instead of training from scratch?**  
Training a model to understand SQL semantics from scratch would require far more than 250 pairs. `defog/sqlcoder-7b` was already pre-trained on 20,000+ SQL queries and understands SQL grammar, join semantics, and aggregation patterns deeply. Fine-tuning with QLoRA adapts only 0.19% of parameters to our specific schema and optimization patterns, preserving the base model's SQL knowledge while specializing it to our domain.

**Why 4-bit QLoRA instead of full fine-tuning?**  
The full sqlcoder-7b model requires ~28 GB VRAM in float16. With 4-bit NF4 quantization and LoRA, it fits in 8 GB VRAM (RTX 5070 Laptop) with no meaningful quality loss for fine-tuning tasks, as shown by the eval loss of 0.084.

**Why use the same classifier for both detection and ranking?**  
The classifier predicts execution time as a continuous value, making it directly usable as a cost function for ranking candidate rewrites. This creates a closed-loop system where the same model that flags a slow query also evaluates the LLM's suggested fixes — without requiring additional ground-truth labels for the ranking stage.