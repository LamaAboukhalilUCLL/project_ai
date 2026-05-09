# AI-Powered SQL Slow-Query Monitor

**Course**: Advanced AI (BCS) [MBI36j]  
**Authors**: Lama Abou Khalil, Lina Belabed 
**Date**: 9 May 2026

---

## 1. Problem

Modern web applications run thousands of SQL queries per minute. Slow queries
degrade user experience invisibly; they don't crash, they just take longer than
they should. We built **a live monitor that detects slow PostgreSQL queries,
predicts how slow they are, generates and ranks candidate optimizations, and
verifies the top candidate against the database** before presenting a structured
response.

The system monitors a Stats Stack Exchange dump (~345k users, 425k posts, 1.7M
votes) and routes any query whose mean execution time exceeds 100ms (100ms is the threshold we use in slow vs fast) through the
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
[6] DB verification — cold + warm EXPLAIN ANALYZE on top candidate
        ↓
[7] Storage — verified wins → embedding history + query_log.json → dashboard
```

The pipeline has seven stages:

1. **Detection** — `pg_stat_statements` is polled every 5 seconds. Queries with
   `mean_exec_time > 100ms` are extracted and deduplicated against past picks.
2. **Feature extraction** — for each candidate query, we compute 20 features
   in two families: 8 lexical patterns derived via regex (e.g. `has_select_star`,
   `join_count`, `has_function_in_where`) and 12 features extracted by walking
   the JSON-formatted output of `EXPLAIN (ANALYZE, BUFFERS)` (planner cost,
   estimated/actual rows, buffer hits/reads, scan and join types).
3. **Discrimination** — a multi-task neural network with a shared 64→32 trunk
   and two heads (classification and regression) decides whether the query
   warrants optimization (`p_slow > 0.5`) and predicts its log execution time.
4. **Memory lookup** — before generating new candidates, the system queries a
   sentence-transformer embedding index of past verified fixes. If a
   semantically similar query has been fixed before (cosine similarity ≥ 0.8),
   the previous fix is provided as context to the LLM.
5. **Generation** — when a query passes the gate, the fine-tuned
   `defog/sqlcoder-7b` model produces up to 3 candidate rewrites via beam
   search (n_beams=4) and two temperature samples (T=0.7, T=1.0). A
   regex-based validator filters malformed outputs before scoring.
6. **Ranking** — each candidate's predicted execution time is computed by the
   *same* multi-task model; candidates are sorted by predicted speedup vs the
   actual `pg_stat_statements` mean time. **This step is what makes the system
   more than text-in/text-out: the trained discriminator evaluates the
   generator's output, rather than the generator's first guess being treated
   as the answer.**
7. **Verification** — the top-ranked candidate is run against the live database
   with cold + warm timing to produce a measured speedup. Genuine wins are
   stored in a sentence-transformer embedding index for retrieval on future
   similar queries and logged to `dashboard/query_log.json`.

---

## 3. AI components

### 3.1 Multi-task neural network (the discriminator)

Inputs: 20 features. Architecture: shared trunk (Linear 20→64 → ReLU →
Dropout 0.3 → Linear 64→32 → ReLU → Dropout 0.3) with two heads (each
Linear 32→16 → ReLU → Linear 16→1). Total trainable parameters: ~3.5k.

**Training objective**:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCE}} + (1 - \alpha) \cdot \mathcal{L}_{\text{quantile}}(\tau=0.7)$$

with α=0.6. Quantile loss with τ=0.7 was chosen after the initial MSE-trained
model showed systematic under-prediction at the high end of the time
distribution. Under-prediction is the worse failure mode for a slow-query
monitor — missing a slow query is more costly than a false positive — so we
biased the regression head upward with an asymmetric loss.

**Training data**: 1,438 queries measured on the Stats Stack Exchange database,
split 80/20 train/test. Label distribution: ~60% slow (warm median > 100ms),
~40% fast. Trained for 150 epochs, batch size 32, learning rate 1e-3,
AdamW optimizer.

**Test set results (n=164 held-out queries)**:

| Metric | Value |
|---|---|
| Classification accuracy | 97.6% |
| Slow-class recall | 1.000 |
| Slow-class precision | 0.968 |
| Regression log-MAE | 0.716 |
| Regression R² (log scale) | 0.645 |

The regression head predicts log-execution-time; predictions are
exponentiated at inference time. A log-MAE of 0.716 corresponds to predictions
being off by an average factor of approximately 2.0× ; sufficient for
distinguishing fast candidates from slow ones in the ranking step.

### 3.2 Fine-tuned query generator (second-stage domain adaptation)

We used `defog/sqlcoder-7b` (7.25 billion parameters) as the base model.
`sqlcoder-7b` was already fine-tuned by Defog.ai on 20,000+ human-curated
SQL queries across 10 schemas, giving it deep SQL semantic knowledge. Our
second-stage fine-tuning adds a domain-specific LoRA adapter that teaches the
model our specific schema (`posts`, `users`, `votes`) and our optimization
patterns.

**Method**: 4-bit NF4 quantization (BitsAndBytes) + LoRA targeting attention
projection layers (`q_proj`, `v_proj`, `k_proj`, `o_proj`), rank r=16,
α=32. Only 13.6M of 7.25B parameters are trained (0.19%), preserving the base
model's SQL knowledge while adapting it to our domain.

**Training data**: 250 (slow query → optimized query) pairs, generated by
GPT-4o-mini with the database schema as context. Pairs cover 12 SQL anti-
patterns: `SELECT *`, correlated subqueries, multi-table JOINs without LIMIT,
`LIKE` wildcards, `ORDER BY` without LIMIT, `UNION`, `CASE WHEN`,
`EXISTS`/`NOT EXISTS`, CTEs, window functions, self-joins, and date/time
aggregations.

**Training results** (3 epochs, batch size 2, gradient accumulation 4,
learning rate 2e-4, cosine schedule, RTX 5070 Laptop GPU, ~10 minutes):

| Epoch | Eval loss |
|---|---|
| 1 | 0.0973 |
| 2 | **0.0841** |
| 3 | 0.0873 |

Best checkpoint at epoch 2 (eval loss 0.0841). The model produces clean,
schema-correct SQL with no tokenizer corruption — a significant improvement
over the earlier `codet5-small` (60M) approach which produced malformed
identifiers such as `u.99` and broken aliases.

### 3.3 Model-in-the-loop ranking

For each candidate rewrite, the same multi-task model that flagged the
original query re-extracts features from the candidate SQL (via `EXPLAIN
ANALYZE`) and queries its regression head to predict the candidate's execution
time. Candidates are sorted by predicted speedup:

$$\text{speedup}_{i} = t_{\text{pg\_stat}} - \hat{t}_{\text{candidate}_i}$$

where $t_{\text{pg\_stat}}$ is the actual mean execution time from
`pg_stat_statements`, not the model's prediction of the original. The top
candidate is verified against the live database; the others are logged but
not executed.

### 3.4 Embedding-based memory

Verified fixes are stored as sentence-transformer embeddings
(`all-MiniLM-L6-v2`, 384-dimensional). Before generating new candidates, the
monitor computes the cosine similarity between the incoming query and all
stored fixes. If the best match exceeds 0.8 similarity, the prior fix is
provided as context to the LLM, steering it toward known-good patterns.
In our live evaluation, 7 of 8 queries had a similar fix found in history
(average similarity 0.91), demonstrating the system's ability to accumulate
and reuse domain knowledge over time.

---

## 4. Evaluation

### 4.1 Discriminator quality

| Metric | Held-out test (n=164) | Manual spot-check (n=20) |
|---|---|---|
| Classification accuracy | 97.6% | 100% |
| Slow-class recall | 1.000 | — |
| Slow-class precision | 0.968 | — |
| Regression log-MAE | 0.716 | 0.716 |
| Regression R² (log scale) | 0.645 | — |

A slow-class recall of 1.000 on the held-out test set means the classifier
missed zero slow queries — the desired property for a monitor where false
negatives are more costly than false positives.

### 4.2 Live pipeline evaluation (8 queries)

We ran the full pipeline on 8 representative slow queries covering the main
anti-patterns in our training data:

| Query pattern | Original (ms) | Fixed (ms) | Saved (ms) | Speedup factor |
|---|---|---|---|---|
| `SELECT *` with `ORDER BY` no `LIMIT` | 404.8 | 0.4 | 404.4 | 1012× |
| Correlated subquery `COUNT` | 709.6 | 512.6 | 197.0 | 1.38× |
| Correlated subquery `AVG` | 967.5 | 567.4 | 400.1 | 1.71× |
| 3-table `JOIN` with `SELECT *` | 4318.3 | 544.1 | 3774.2 | 7.94× |
| `LIKE` wildcard with `OR` | 147.4 | — | — | skipped† |
| `SELECT *` `ORDER BY` unindexed | 254.2 | 46.9 | 207.3 | 5.42× |
| Multiple correlated subqueries | 539.8 | 434.5 | 105.3 | 1.24× |
| `DISTINCT` with 3-table `JOIN` | 672.5 | 290.9 | 381.6 | 2.31× |

+ The `LIKE` query was correctly skipped by the classifier (p_slow = 9.7%,
below the 0.5 threshold), the planner features indicated the query was not
actually slow despite its textual appearance.

**Aggregate results (7 attempted queries):**

| Metric | Value |
|---|---|
| Win rate | 7/7 = 100% |
| Total time saved | 5,469.9ms (5.47s) |
| Average speedup | 781.4ms per query |
| Median speedup | 381.6ms per query |
| Best single fix | 3,774ms (3-table JOIN) |
| Classifier correct skip rate | 1/1 = 100% |

### 4.3 Defense of design decisions

- **Why a custom multi-task model and not a single regression head?** Two heads
  share the same 20-feature representation while serving different downstream
  uses: the classification head gates the optimization pipeline with a
  calibrated probability, while the regression head provides a continuous
  score for ranking candidates. A single regression head would require a
  separate threshold-tuning step for the gating decision.

- **Why plan features when text features were already there?** The `LIKE`
  query in our evaluation demonstrates this concretely: text features flagged
  it as likely slow (`has_like_wildcard=1`, `has_or=1`), but the plan features
  showed `plan_total_cost=0` and no sequential scan — the planner was using an
  index. The combined model correctly classified it as fast (p_slow=9.7%),
  while a text-only model would have triggered the full optimization pipeline
  unnecessarily.

- **Why second-stage fine-tuning instead of training a generator from scratch?**
  Training a model to understand SQL semantics from 250 examples would produce
  a model that memorises patterns rather than generalising. `defog/sqlcoder-7b`
  already understands SQL grammar, join semantics, and aggregation patterns from
  its 20,000-query pre-training. Fine-tuning with QLoRA adapts only 0.19% of
  parameters to our specific schema, preserving the base model's SQL competence
  while specialising it to our domain.

- **Why use the discriminator for ranking rather than a separate ranker?**
  Training a separate ranking model would require labelled pairs of
  (slow query, candidate rewrites) with ground-truth speedup measurements —
  data we do not have at scale. The regression head of the discriminator
  provides a zero-cost proxy for ranking: it was trained to predict execution
  time, so lower predicted time naturally corresponds to a better candidate.

---

## 5. Your feedback addressed (Last class of Advanced AI)

| Concern | How addressed |
|---|---|
| **Project too simple, lacks advanced AI** | Replaced single-task MLP on 8 regex features with a multi-task neural network on 20 features (text + planner-derived). Added second-stage QLoRA fine-tuning of a 7B SQL-specialist model and a learned neural ranker. The pipeline integrates four ML components: discriminator, generator, ranker, embedding-based memory. |
| **Training dataset too small** | Grew from 465 to 1,438 queries (+209%) using parameterised template-based generation covering 12 SQL anti-pattern categories with random parameter sampling to ensure diversity across runs. |
| **Execution times unreliable** | Replaced single hot-cache run with cold (post-`DISCARD ALL`) + warm (median of multiple runs) measurements; adaptive run count (5 warm runs for queries under 500ms, 2 for 500ms–2s, 1 for >2s); explicit handling of 10s timeouts (capped at 10,000ms, not dropped). |
| **Avoid pure text-in/text-out** | Pipeline does not return the generator's first guess. The trained discriminator scores each candidate's predicted execution time, ranks them by predicted speedup against the actual measured baseline, and the top candidate is verified against the live database before being stored. The decision-making component is the discriminator + verifier loop, not the generator. |

---

## 6. Limitations and future work

- The 100ms slow/fast threshold is a hard cutoff; queries near 100ms are
  inherently ambiguous and inflate misclassification counts at the boundary.
  A soft margin or probability-based threshold would be more principled.
- The 10-second `statement_timeout` creates an artificial ceiling in the
  training data: queries that hit the cap are recorded as exactly 10,000ms,
  distorting the regression target by collapsing the upper tail to a single
  value. A follow-up iteration should exclude timed-out queries from the
  regression target while retaining them for the classification head.
- The fine-tuned generator was trained on outputs from GPT-4o-mini, so it
  inherits that model's biases and cannot exceed its quality ceiling. An
  ensemble with rule-based candidates (e.g. automatic index suggestions,
  `SELECT *` expansion) would add diversity without this limitation.
- All measurements are from a single Stack Exchange dump on one machine.
  Execution times are hardware-dependent; the classifier would require
  retraining on a different machine or database to maintain calibration.
- The regression head predicts absolute execution time, which is
  hardware-dependent. A relative speedup prediction (candidate / original)
  would generalise better across hardware configurations.

---

## 7. Reproducibility

The full pipeline is reproducible from a fresh clone with two Python
environments (Python 3.13 for the project runtime, Python 3.12 + CUDA for
model training):

```
# 1. Database setup
psql -f schema.sql
python loading_data/local_data.py

# 2. Training data collection
python queries_generation/all_queries.py          # ~35 minutes
python queries_generation/generate_queries.py     # additional diverse queries

# 3. Classifier training
python classifier/classifier.py

# 4. LLM fine-tuning (requires venv312 + GPU)
python finetune/build_dataset.py                  # generate pairs via GPT-4o-mini
python finetune/finetune_sqlcoder.py              # QLoRA fine-tune (~10 minutes on my GPU, will be different for you)

# 5. Run the system
python monitor/monitor.py                         # live pipeline, you write queries in pgAdmin, and you watch our model either in the terminal or the dashboard doing its work
python dashboard/app.py                           # dashboard at localhost:5000, provides graphs at the end too, index.html is an ai-generated template, but that does not matter
```

See `README.md` for full setup details including environment variables and
GPU configuration for RTX 40/50 series (requires PyTorch nightly for sm_120
Blackwell architecture).