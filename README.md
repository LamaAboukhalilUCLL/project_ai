## Setup

### Prerequisites

- Python 3.12+
- PostgreSQL 16+ (this project was developed against PostgreSQL 18)
- ~500 MB free disk space for the Stack Exchange data dump

### 1. Install Python dependencies

```powershell
pip install -r requirements.txt
```

### 2. Set up the database

Create the database and schema:

```powershell
psql -U postgres -h localhost -c "CREATE DATABASE stackexchange_db;"
psql -U postgres -h localhost -d stackexchange_db -f schema.sql
```

The default credentials assumed by the code are `postgres / postgres` on `localhost:5432`. If yours differ, update the `DB_CONFIG` blocks in `monitor/monitor.py`, `loading_data/local_data.py`, and `queries_generation/all_queries.py`.

### 3. Load the Stack Exchange data

Download the `ai.stackexchange.com.7z` dump from the [Stack Exchange Data Dump on archive.org](https://archive.org/details/stackexchange) and extract `Users.xml`, `Posts.xml`, `Votes.xml` to the project root.

Then load:

```powershell
python loading_data/local_data.py
```

This takes a few minutes. Verify with:

```powershell
psql -U postgres -h localhost -d stackexchange_db -c "SELECT (SELECT COUNT(*) FROM users) AS users, (SELECT COUNT(*) FROM posts) AS posts, (SELECT COUNT(*) FROM votes) AS votes;"
```

You should see roughly 71k users, 27k posts, 93k votes.

## Generating training data

The script `queries_generation/all_queries.py` runs ~465 hand-crafted SQL queries against the loaded database and records timing and query-plan features for each one.

```powershell
python queries_generation/all_queries.py
```

Runtime is ~35 minutes. The output is `training_data.csv` at the project root.

### Methodology

For each query, we measure:

- **Cold-cache time** (run 1, after `DISCARD ALL`)
- **Warm-cache median time** (median of 2–5 subsequent runs)
- **Warm-cache standard deviation** (stability indicator)
- **All raw run times** (kept for transparency)
- **Plan features** (extracted from `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`):
  estimated cost, planned and actual row counts, plan tree depth,
  shared buffer hits and reads, and presence of sequential scans, index scans,
  bitmap scans, hash joins, nested loops, and merge joins.

The number of warm runs is **adaptive**: queries faster than 500ms are run 5 times warm for a stable median; queries between 500ms and 2s are run 2 times warm; queries over 2s or that hit the 10-second statement timeout are run only once warm to keep total runtime reasonable.

A query is labelled `slow` if its warm median exceeds 100ms, else `fast`. Queries that hit the 10s timeout are recorded with `execution_time = 10000ms` and a non-zero `timed_out_runs` count, so the model treats their times as a lower bound rather than discarding them.

## Project structure

```
project_ai/
├── classifier/          # Multi-task neural model (slow/fast + execution time)
├── dashboard/           # Flask dashboard for visualizing flagged queries
├── embeddings/          # Sentence-transformer history of past fixes
├── loading_data/        # Stack Exchange XML → PostgreSQL loader
├── monitor/             # Live slow-query monitor + GPT optimization loop
├── queries_generation/  # Training-data generation script
├── schema.sql           # Database schema
├── training_data.csv    # Generated training data
└── requirements.txt
```