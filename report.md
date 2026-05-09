# AI-Powered SQL Slow-Query Monitor and Optimizer

**Course**: Advanced AI (BCS) [MBI36j]  
**Authors**: Lama Abou Khalil, Lina Belabed  
**Repository**: https://github.com/LamaAboukhalilUCLL/project_ai  
**Date**: 9 May 2026

---

## 1. Problem

Modern web applications run thousands of SQL queries per minute. Slow queries
degrade user experience invisibly; they don't crash, they just take longer than
they should. We built **a live monitor that detects slow PostgreSQL queries,
predicts how slow they are, generates and ranks candidate optimizations, and
verifies the top candidate against the database** before presenting a structured
response.

The system monitors a Stats Stack Exchange dump https://ia800508.us.archive.org/view_archive.php?archive=/30/items/stackexchange/stats.stackexchange.com.7z (~345k users, 425k posts, 1.7M
votes) and routes any query whose mean execution time exceeds 100ms through the
AI pipeline below.

---

## 2. System architecture

```
pg_stat_statements enabled from pgAdmin (poll every 5s)
        ↓
[1] Feature extraction — 8 lexical + 12 plan = 20 features
        ↓
[2] Multi-task neural classifier — p_slow + predicted_ms
        ↓  (if p_slow ≥ 0.5)
[3] Embedding lookup — similar past fix retrieval (all-MiniLM-L6-v2)
        ↓
[4] Fine-tuned LLM — defog/sqlcoder-7b + QLoRA adapter (3 candidates)
        ↓
[5] Neural ranker — same classifier re-scores candidates by predicted speedup
        ↓
[6] DB verification — cold + warm timing on top candidate
        ↓
[7] Storage — verified wins → embedding history + query_log.json → dashboard
```

The pipeline has seven stages:

1. **Detection** — `pg_stat_statements` is polled every 5 seconds. Queries with
   `mean_exec_time > 100ms` are extracted and deduplicated against past picks.
2. **Feature extraction** — 20 features in two families: 8 lexical patterns
   (regex-derived: `has_select_star`, `join_count`, `has_function_in_where`, etc.)
   and 12 features from `EXPLAIN (ANALYZE, BUFFERS)` (planner cost, estimated/actual
   rows, buffer hits/reads, scan and join types).
3. **Discrimination** — a multi-task neural network decides whether the query
   warrants optimization (`p_slow > 0.5`) and predicts its log execution time.
4. **Memory lookup** — sentence-transformer embedding index of past verified fixes.
   If a semantically similar query has been fixed before (cosine similarity ≥ 0.8),
   the prior fix is provided as context to the LLM.
5. **Generation** — the fine-tuned `defog/sqlcoder-7b` produces up to 3 candidate
   rewrites via beam search (n_beams=4) and two temperature samples (T=0.7, T=1.0).
   A regex-based validator filters malformed outputs before scoring.
6. **Ranking** — the same multi-task model re-scores each candidate via EXPLAIN
   ANALYZE and sorts by predicted speedup against the actual `pg_stat_statements`
   mean time. **This is what makes the system more than text-in/text-out: the
   trained discriminator evaluates the generator's output.**
7. **Verification** — the top candidate is run against the live database with
   cold + warm timing. Genuine wins are stored in the embedding index and logged
   to `dashboard/query_log.json`.

---

## 3. AI components

### 3.1 Multi-task neural network (the discriminator)

Inputs: 20 features. Architecture: shared trunk (Linear 20→64 → ReLU →
Dropout 0.3 → Linear 64→32 → ReLU → Dropout 0.3) with two heads (each
Linear 32→16 → ReLU → Linear 16→1). Total trainable parameters: ~3.5k.

**Training objective**:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCE}} + (1 - \alpha) \cdot \mathcal{L}_{\text{quantile}}(\tau=0.7)$$

with α=0.6. Quantile loss with τ=0.7 was chosen over MSE because
under-prediction is the worse failure mode for a monitor — missing a slow
query is more costly than a false positive — so we biased the regression head
upward with an asymmetric loss.

**Training data**: 1,438 queries measured on the Stats Stack Exchange database,
split 80/20 train/test. Label distribution: ~60% slow (warm median > 100ms),
~40% fast. Trained for 150 epochs, batch size 32, learning rate 1e-3, AdamW.

**Test set results (n=284 held-out queries)**:

| Metric | Value |
|---|---|
| Classification accuracy | 97.6% |
| Slow-class recall | 1.000 |
| Slow-class precision | 0.968 |
| Regression log-MAE | 0.716 (~2× avg error) |
| Regression R² (log scale) | 0.645 |

### 3.2 Fine-tuned query generator (second-stage domain adaptation)

We used `defog/sqlcoder-7b` (7.25B parameters) as the base model. It was
already fine-tuned by Defog.ai on 20,000+ human-curated SQL queries, giving it
deep SQL semantic knowledge. Our second-stage fine-tuning adds a domain-specific
LoRA adapter for our schema (`posts`, `users`, `votes`) and optimization patterns.

**Method**: 4-bit NF4 quantization (BitsAndBytes) + LoRA on attention projection
layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`), rank r=16, α=32. Only 13.6M
of 7.25B parameters are trained (0.19%).

**Training data**: 250 (slow → optimized) SQL pairs generated by GPT-4o-mini,
covering 12 anti-patterns: `SELECT *`, correlated subqueries, multi-table JOINs,
`LIKE` wildcards, `ORDER BY` without LIMIT, `UNION`, `CASE WHEN`, `EXISTS`/`NOT EXISTS`,
CTEs, window functions, self-joins, and date aggregations.

**Training results** (3 epochs, RTX 5070 Laptop GPU, ~10 minutes):

| Epoch | Eval loss |
|---|---|
| 1 | 0.0973 |
| 2 | **0.0841** |
| 3 | 0.0873 |

Best checkpoint at epoch 2 (eval loss 0.0841). The model produces clean,
schema-correct SQL — a significant improvement over the earlier `codet5-small`
(60M) approach which produced malformed identifiers such as `u.99`.

### 3.3 Model-in-the-loop ranking

Each candidate's predicted execution time is computed by the same classifier
via EXPLAIN ANALYZE. Candidates are sorted by:

$$\text{speedup}_{i} = t_{\text{pg\_stat}} - \hat{t}_{\text{candidate}_i}$$

where $t_{\text{pg\_stat}}$ is the actual measured baseline, not the model's
prediction of the original query.

### 3.4 Embedding-based memory

Verified fixes are stored as sentence-transformer embeddings
(`all-MiniLM-L6-v2`, 384-dimensional). In our live evaluation, 7 of 8 queries
had a similar fix found in history (avg similarity 0.91), demonstrating the
system's ability to accumulate and reuse domain knowledge over time.

---

## 4. Evaluation

### 4.1 Discriminator quality: evaluations/evaluation_sample.csv and evaluations/evaluation.png

| Metric | Held-out test (n=284) | Spot-check (n=20) |
|---|---|---|
| Classification accuracy | 97.6% | 85% |
| Slow-class recall | 1.000 | — |
| Slow-class precision | 0.968 | — |
| Regression log-MAE | 0.716 | 0.610 |
| Regression R² (log scale) | 0.645 | — |

Slow-class recall of 1.000 means the classifier missed zero slow queries on
the held-out test set. R²=0.645 is sufficient for the ranking task — relative
ordering of candidates matters, not absolute prediction accuracy.

### 4.2 Controlled pipeline comparison (16 queries): evaluations/evaluation_pipeline.csv and evaluations/evaluation_pipeline.png

We ran a controlled evaluation on 16 slow queries sampled from the training
dataset (excluding timed-out queries), comparing GPT-only against the full
pipeline on real DB execution:

| Approach | Win rate | Avg speedup |
|---|---|---|
| GPT-only | 62.5% | −11.9% |
| **Full pipeline** | **75.0%** | **+18.1%** |

The full pipeline outperforms GPT-only by +12.5pp on win rate. Notably,
GPT-only averages negative speedup — it makes GROUP BY aggregations slower
because the PostgreSQL planner already optimizes them well. The full pipeline's
classifier ranking filters these bad suggestions.

### 4.3 Live monitor verification (8 queries): evaluations/results_of_model_in_one_run/one_run_results.json

| Query pattern | Original (ms) | Fixed (ms) | Saved (ms) | Factor |
|---|---|---|---|---|
| `SELECT *` + `ORDER BY` no LIMIT | 404.8 | 0.4 | 404.4 | 1012× |
| 3-table JOIN `SELECT *` | 4318.3 | 544.1 | 3774.2 | 7.94× |
| Correlated subquery AVG | 967.5 | 567.4 | 400.1 | 1.71× |
| DISTINCT 3-table JOIN | 672.5 | 290.9 | 381.6 | 2.31× |
| `LIKE` wildcard | 147.4 | — | — | skipped† |

The LIKE query was correctly skipped (p_slow=9.7%) — plan features showed an
index scan despite the wildcard, demonstrating the value of combining text and
plan features. 7/7 attempted queries verified faster.

### 4.4 Defense of design decisions

- **Why multi-task instead of single regression?** The classification head gates
  the pipeline with a calibrated probability; the regression head ranks candidates.
  A single head cannot serve both purposes cleanly.

- **Why plan features?** The LIKE query proves their value: text features flagged
  it as slow, but plan features showed zero sequential scan. Text-only would have
  wasted the full pipeline on a fast query.

- **Why second-stage fine-tuning?** 250 pairs cannot teach SQL semantics from
  scratch. `sqlcoder-7b` already knows SQL deeply; we only adapt it to our schema.

- **Why use the discriminator for ranking?** Training a separate ranker requires
  ground-truth speedup pairs we don't have. The regression head is a zero-cost proxy.

---

## 5. Feedback addressed

| Concern | How addressed |
|---|---|
| **Too simple, lacks advanced AI** | Multi-task neural network + QLoRA fine-tuning of 7B model + neural ranker + embedding memory — four integrated AI components. |
| **Dataset too small** | Grew from 465 to 1,438 queries (+209%) using parameterised template generation covering 12 anti-pattern categories. |
| **Execution times unreliable** | Cold cache flush + warm median (2–5 runs); adaptive run count; explicit timeout handling at 10,000ms. |
| **Pure text-in/text-out** | The discriminator scores every candidate and ranks by predicted speedup; top candidate verified on DB before storing. Generator output is never returned unverified. |

---

## 6. Contributions

**Lama Abou Khalil:**
- Full pipeline architecture and `monitor/monitor.py`
- Multi-task neural classifier (design, feature engineering, training)
- QLoRA fine-tuning of defog/sqlcoder-7b
- Flask dashboard with Chart.js visualizations
- Training data collection (`all_queries.py`, `generate_queries.py`)
- Evaluation scripts and analysis

**Lina Belabed:**
- Database setup and Stack Exchange XML data loading
- Fine-tuning pair generation (`build_dataset.py`)
- Embedding-based memory component (`embeddings.py`)
- `classifier.ipynb` notebook and training visualizations
- Report writing

**External resources used:**
- `defog/sqlcoder-7b` (HuggingFace) — pre-trained SQL model, used as fine-tuning base
- QLoRA paper (Dettmers et al., 2023) — quantization + LoRA methodology
- `sentence-transformers` library — embedding similarity search
- `bitsandbytes` library — 4-bit quantization implementation

**GenAI usage:**
- GPT-4o-mini: generated the 250 fine-tuning pairs (slow → optimized SQL)
- GPT-4o-mini: used as the baseline in the pipeline comparison evaluation
- Claude: debugging assistance, code review and assistance during development, explanation of new concepts, and enhancing the writing of this report

---

## 7. Limitations and future work

**Challenges:**

- RTX 5070 Laptop that the main setup was done on uses Blackwell architecture (sm_120) — not supported by stable PyTorch, requiring nightly CUDA builds and Python 3.12 specifically.
- Execution time non-determinism: the same query can vary ±30% between runs due to caching, making labels noisy near the 100ms boundary (which is why we run SELECT pg_stat_statements_reset(); in PgAdmin before each test-run)
- The regression head is capped at 10,000ms due to statement_timeout, distorting the upper tail of the training distribution.
- We tried a lot of previous models, t5-small (this model knows nothing about coding, turned out to perform very badly given all the time and retraining done on it, due to the data limitation and the fact that it is text-based), codet5(small or base, did systematical errors, did not learn our database, and it was not trained well on sql). Which pushed us to use our current model, but fine-tune it again and adapt it to our own usage (still a much better alternative than our first gpt approach which was simple)
- The idea of the project turned out a bit complex to actually develop, a lot of setup, a lot of new unlearned strategies which required us to learn them first and try to understand what we are doing at every point, we were hopefull to train an LLM from scratch, but that would've required a lot of time and resources given the time limitations

**Future work: (may or may not be done)**

- Replace the hard 100ms threshold with a soft probability gate or learned threshold
- Add rule-based candidates (automatic index suggestions, column expansion) alongside the LLM to increase diversity
- Train the regression head on relative speedup (candidate/original) rather than absolute time for better hardware generalization
- Extend to multi-tenant monitoring with per-schema fine-tuning adapters
- Enhance the Dashboard
- Try generalizing it on many other databases

---

## 8. Reproducibility (check README.md)

```bash
# 1. Database setup (requires PostgreSQL 16+)
psql -U postgres -f schema.sql
python loading_data/local_data.py

# 2. Training data collection (~35 minutes)
python queries_generation/all_queries.py
python queries_generation/generate_queries.py

# 3. Classifier training
python classifier/classifier.py

# 4. Download fine-tuned model adapter (~800MB)
huggingface-cli download lamahugface/sqlcoder-stackexchange \
  --local-dir finetune/sqlcoder-finetuned

# OR retrain from scratch (~10 min on RTX GPU, requires venv with CUDA)
python finetune/build_dataset.py
python finetune/finetune_sqlcoder.py

# 5. Run
python monitor/monitor.py   # live pipeline (terminal output)
python dashboard/app.py     # dashboard at http://localhost:5000
```

See `README.md` for full environment setup. Requires Python 3.12 with CUDA
nightly PyTorch (RTX 40/50 series GPU, sm_89+/sm_120).