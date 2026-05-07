"""
Live SQL slow-query monitor with model-in-the-loop optimization.

Pipeline:
  1. Poll pg_stat_statements for recent slow queries.
  2. For each: QueryAnalyzer predicts p_slow + predicted_ms + plan features.
  3. If predicted slow, ask GPT for K candidate rewrites (not just 1).
  4. Rank candidates by predicted speedup using the same model.
  5. Verify the top candidate's actual time with a cold + warm measurement.
  6. Log a structured response (original, predicted, ranked alternatives,
     measured speedup) for the dashboard.
  7. Store genuine wins in the embeddings history.
"""

import time
import json
import os
import re
import sys
import datetime
import statistics

import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.inference import QueryAnalyzer, replace_placeholders
from embeddings.embeddings import find_similar, store_fix


# ─────────────────────────── Config ───────────────────────────

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_CONFIG = {
    "dbname": "stackexchange_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432",
}

P_SLOW_THRESHOLD = 0.5      # only optimize queries the model thinks are slow
POLL_INTERVAL_SECONDS = 5
N_CANDIDATES = 3            # how many alternatives we ask GPT for
LOG_FILE = "dashboard/query_log.json"


# ─────────────────────────── Schema helper ───────────────────────────

def get_table_schema(conn):
    cur = conn.cursor()
    schema = ""
    for table in ["posts", "users", "votes"]:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = %s
            ORDER BY ordinal_position
        """, (table,))
        cols = cur.fetchall()
        schema += f"\nTable: {table}\n"
        for col_name, col_type in cols:
            schema += f"  - {col_name}: {col_type}\n"
    cur.close()
    return schema


# ─────────────────────────── Candidate generation ───────────────────────────

def ask_gpt_for_candidates(slow_query, schema, similar_fix=None, n=N_CANDIDATES):
    """
    Ask GPT for multiple candidate rewrites. We ask for several because
    the model-in-the-loop ranker can pick the best one — relying on a
    single GPT suggestion means we have nothing to rank against.
    """
    past_example = ""
    if similar_fix:
        past_example = (
            f"\nA semantically similar query was optimized before:\n"
            f"Slow: {similar_fix['slow_query']}\n"
            f"Fix: {similar_fix['fix']}\n"
            f"Speedup: {similar_fix['speedup_ms']}ms\n"
        )

    prompt = f"""You are a PostgreSQL performance expert.
A slow SQL query was detected. Propose {n} different optimized versions.

Database schema:{schema}

Slow query:
{slow_query}
{past_example}
Respond ONLY with a JSON object of this exact shape:
{{
  "reason": "one sentence on why the query is slow",
  "candidates": [
    {{"sql": "...", "strategy": "short label (e.g. 'add LIMIT', 'rewrite IN as JOIN')"}},
    {{"sql": "...", "strategy": "..."}},
    ...
  ],
  "index_suggestion": "CREATE INDEX ... or NONE"
}}

Each candidate must be a complete, runnable SELECT. Do not include explanations
outside the JSON. Do not wrap the JSON in markdown fences."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Best-effort recovery if GPT slips and adds prose.
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        return json.loads(m.group(0)) if m else {
            "reason": "(parse error)",
            "candidates": [],
            "index_suggestion": "NONE",
        }


# ─────────────────────────── Verification (cold + warm) ───────────────────────────

def measure_query(conn, query, runs=3, timeout_ms=10000):
    """
    Cold + warm measurement for verifying a candidate. Lighter than the
    training-time methodology (3 runs total, not 6) because we're verifying
    one candidate, not building a labelled dataset.
    """
    safe = replace_placeholders(query)
    cur = conn.cursor()
    times = []
    try:
        cur.execute(f"SET statement_timeout = '{timeout_ms}ms'")
        cur.execute("DISCARD ALL")
        for _ in range(runs):
            try:
                cur.execute(f"EXPLAIN ANALYZE {safe}")
                for row in cur.fetchall():
                    if "Execution Time:" in row[0]:
                        m = re.search(r"Execution Time: ([\d.]+)", row[0])
                        if m:
                            times.append(float(m.group(1)))
                            break
            except psycopg2.errors.QueryCanceled:
                times.append(float(timeout_ms))
    except Exception as e:
        print(f"  [measure error: {e}]")
        return None
    finally:
        cur.close()

    if not times:
        return None
    return {
        "cold_ms": times[0],
        "warm_median_ms": statistics.median(times[1:]) if len(times) > 1 else times[0],
        "all_runs_ms": times,
    }


# ─────────────────────────── pg_stat_statements polling ───────────────────────────

def get_recent_slow_queries(conn, threshold_ms=100):
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT query, mean_exec_time, calls
            FROM pg_stat_statements
            WHERE mean_exec_time > %s
              AND query NOT LIKE '%%pg_stat_statements%%'
              AND query NOT LIKE '%%EXPLAIN%%'
              AND query NOT LIKE '%%SET%%'
              AND query ILIKE '%%SELECT%%'
            ORDER BY mean_exec_time DESC
            LIMIT 10
        """, (threshold_ms,))
        return cur.fetchall()
    except Exception as e:
        conn.rollback()
        print(f"Error reading pg_stat_statements: {e}")
        return []
    finally:
        cur.close()


# ─────────────────────────── Logging ───────────────────────────

def log_event(event):
    log = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r") as f:
                log = json.load(f)
        except json.JSONDecodeError:
            log = []
    log.append(event)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


# ─────────────────────────── Main loop ───────────────────────────

def process_one_query(query, mean_time_ms, calls, analyzer, conn, schema):
    """Run the full pipeline on a single detected slow query."""
    print(f"\n[CANDIDATE] {query[:100]}{'...' if len(query) > 100 else ''}")
    print(f"  pg_stat mean time: {mean_time_ms:.1f}ms ({calls} calls)")

    # 1. Model analyzes the original query
    baseline = analyzer.analyze(query, conn=conn)
    print(f"  model: p_slow={baseline['p_slow']*100:.1f}%  "
          f"predicted={baseline['predicted_ms']:.1f}ms  "
          f"cost={baseline['features']['plan_total_cost']:.0f}")

    if baseline["p_slow"] < P_SLOW_THRESHOLD:
        print(f"  -> below threshold ({P_SLOW_THRESHOLD}); skipping")
        return None

    # 2. Look for a semantically similar past fix
    similar = find_similar(query)

    # 3. Generate candidate rewrites
    print(f"  asking GPT for {N_CANDIDATES} candidates...")
    gpt = ask_gpt_for_candidates(query, schema, similar)
    candidates = [c["sql"] for c in gpt.get("candidates", []) if c.get("sql")]
    if not candidates:
        print("  -> no candidates returned; logging and skipping")
        log_event({
            "type": "no_candidates",
            "query": query[:300],
            "p_slow": baseline["p_slow"],
            "predicted_ms": baseline["predicted_ms"],
            "reason": gpt.get("reason", ""),
            "timestamp": datetime.datetime.now().isoformat(),
        })
        return None

    # 4. Model-in-the-loop ranking
    _, ranked = analyzer.rank_candidates(query, candidates, conn=conn)
    print(f"  ranked {len(ranked)} candidates by predicted speedup:")
    for i, r in enumerate(ranked, 1):
        print(f"    [{i}] predicted={r['predicted_ms']:.1f}ms  "
              f"(speedup vs original: {r['predicted_speedup_ms']:+.1f}ms)")

    top = ranked[0]
    if top["predicted_speedup_ms"] <= 0:
        print("  -> no candidate predicted faster than original; skipping verify")
        verified = None
    else:
        # 5. Verify the top candidate against the DB
        print(f"  verifying top candidate against DB...")
        verified = measure_query(conn, top["query"])
        if verified:
            print(f"  measured: cold={verified['cold_ms']:.1f}ms  "
                  f"warm_median={verified['warm_median_ms']:.1f}ms")

    # Map candidate sql -> strategy label for the log
    strat_by_sql = {c["sql"]: c.get("strategy", "")
                    for c in gpt.get("candidates", [])}

    # 6. Structured log entry
    measured_speedup_ms = None
    if verified:
        measured_speedup_ms = mean_time_ms - verified["warm_median_ms"]

    event = {
        "type": "slow_query_handled",
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query[:500],
        "pg_stat_mean_ms": round(mean_time_ms, 2),
        "calls": calls,
        "model": {
            "p_slow": round(baseline["p_slow"], 4),
            "predicted_ms": round(baseline["predicted_ms"], 2),
            "plan_total_cost": baseline["features"]["plan_total_cost"],
            "has_seq_scan": baseline["features"]["has_seq_scan"],
            "has_index_scan": baseline["features"]["has_index_scan"],
        },
        "reason": gpt.get("reason", ""),
        "index_suggestion": gpt.get("index_suggestion", "NONE"),
        "similar_fix_used": bool(similar),
        "candidates_ranked": [
            {
                "sql": r["query"][:500],
                "strategy": strat_by_sql.get(r["query"], ""),
                "predicted_ms": round(r["predicted_ms"], 2),
                "predicted_speedup_ms": round(r["predicted_speedup_ms"], 2),
            }
            for r in ranked
        ],
        "top_candidate_verified": verified,
        "measured_speedup_ms": (round(measured_speedup_ms, 2)
                                if measured_speedup_ms is not None else None),
    }
    log_event(event)

    # 7. Store the win in embeddings history if it's actually faster
    if (verified and measured_speedup_ms is not None
            and measured_speedup_ms > 0):
        store_fix(query, top["query"], measured_speedup_ms)
        print(f"  stored in embeddings history "
              f"({measured_speedup_ms:.1f}ms saved)")

    return event


def main():
    print("=" * 60)
    print("  SQL Query Monitor — model-in-the-loop pipeline")
    print("=" * 60)

    print("Loading model...")
    analyzer = QueryAnalyzer()
    print(f"  loaded ({analyzer.meta['n_features']}-feature multi-task model)")

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    print("  connected")

    schema = get_table_schema(conn)
    seen = set()

    print(f"\nMonitoring (p_slow threshold: {P_SLOW_THRESHOLD}, "
          f"poll: {POLL_INTERVAL_SECONDS}s)")
    print("-" * 60)

    while True:
        try:
            slow_rows = get_recent_slow_queries(conn)
            for query, mean_time, calls in slow_rows:
                key = query[:100]
                if key in seen:
                    continue
                seen.add(key)
                try:
                    process_one_query(query, mean_time, calls,
                                       analyzer, conn, schema)
                except Exception as e:
                    print(f"  [pipeline error on this query: {e}]")
            time.sleep(POLL_INTERVAL_SECONDS)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[loop error: {e}] continuing in {POLL_INTERVAL_SECONDS}s")
            time.sleep(POLL_INTERVAL_SECONDS)

    conn.close()


if __name__ == "__main__":
    main()