"""
Live SQL slow-query monitor with model-in-the-loop optimization.

Pipeline:
  1. Poll pg_stat_statements for recent slow queries.
  2. For each: QueryAnalyzer predicts p_slow + predicted_ms + plan features.
  3. If predicted slow, use the fine-tuned CodeT5 LLM for K candidate rewrites.
  4. Rank candidates by predicted speedup using the same neural model.
  5. Verify the top candidate's actual time with a cold + warm measurement.
  6. Log a structured response (original, predicted, ranked alternatives,
     measured speedup) for the dashboard.
  7. Store genuine wins in the embeddings history.

Change from previous version:
  ask_gpt_for_candidates() replaced by LLMOptimizer.suggest_candidates()
  which uses the fine-tuned CodeT5 model. Falls back to GPT automatically
  if the fine-tuned model is not yet available.
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.inference import QueryAnalyzer, replace_placeholders
from embeddings.embeddings import find_similar, store_fix
from finetune.llm_optimizer import LLMOptimizer


# ─────────────────────────── Config ───────────────────────────

load_dotenv()

DB_CONFIG = {
    "dbname": "stackexchange_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432",
}

P_SLOW_THRESHOLD = 0.5
POLL_INTERVAL_SECONDS = 5
N_CANDIDATES = 3
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

def get_candidates(optimizer, slow_query, schema, similar_fix=None, n=N_CANDIDATES):
    """
    Generate candidate rewrites using the fine-tuned CodeT5 LLM.
    Falls back to GPT automatically via LLMOptimizer if model not found.

    Returns a dict matching the old GPT structure so the rest of the
    pipeline doesn't need to change:
      {
        "reason": str,
        "candidates": [{"sql": str, "strategy": str}, ...],
        "index_suggestion": str,
        "llm_source": str,
      }
    """
    # If a similar past fix exists, add it as context hint
    context = schema
    if similar_fix:
        context += (
            f"\n\nA similar slow query was previously fixed:\n"
            f"Slow: {similar_fix['slow_query']}\n"
            f"Fix: {similar_fix['fix']}\n"
            f"Speedup: {similar_fix['speedup_ms']}ms\n"
        )

    raw_candidates = optimizer.suggest_candidates(slow_query, schema=context, n=n)

    # Filter to only valid SQL outputs
    valid = [c for c in raw_candidates if c.get("valid", False)]
    if not valid:
        # Fall back to single suggestion if diverse beam search gave nothing valid
        single = optimizer.suggest(slow_query, schema=context)
        valid = [single] if single.get("valid") else []

    # Build reason string from strategy labels
    strategies = [c.get("strategy", "") for c in valid]
    reason = f"Query optimization strategies: {', '.join(s for s in strategies if s)}"

    candidates = [
        {"sql": c["optimized_sql"], "strategy": c.get("strategy", "")}
        for c in valid
    ]

    llm_source = raw_candidates[0].get("source", "unknown") if raw_candidates else "unknown"

    return {
        "reason": reason,
        "candidates": candidates,
        "index_suggestion": "NONE",
        "llm_source": llm_source,
    }


# ─────────────────────────── Verification (cold + warm) ───────────────────────────

def measure_query(conn, query, runs=3, timeout_ms=10000):
    """
    Cold + warm measurement for verifying a candidate.
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


# ─────────────────────────── Main pipeline ───────────────────────────

def process_one_query(query, mean_time_ms, calls, analyzer, optimizer, conn, schema):
    """Run the full pipeline on a single detected slow query."""
    print(f"\n[CANDIDATE] {query[:100]}{'...' if len(query) > 100 else ''}")
    print(f"  pg_stat mean time: {mean_time_ms:.1f}ms ({calls} calls)")

    # 1. Neural model analyzes the original query
    baseline = analyzer.analyze(query, conn=conn)
    print(f"  model: p_slow={baseline['p_slow']*100:.1f}%  "
          f"predicted={baseline['predicted_ms']:.1f}ms  "
          f"cost={baseline['features']['plan_total_cost']:.0f}")

    if baseline["p_slow"] < P_SLOW_THRESHOLD:
        print(f"  -> below threshold ({P_SLOW_THRESHOLD}); skipping")
        return None

    # 2. Look for a semantically similar past fix
    similar = find_similar(query)

    # 3. Generate candidate rewrites with fine-tuned LLM
    print(f"  generating {N_CANDIDATES} candidates with LLM ({optimizer._device if not optimizer._use_fallback else 'GPT fallback'})...")
    llm_result = get_candidates(optimizer, query, schema, similar, n=N_CANDIDATES)
    candidates = [c["sql"] for c in llm_result.get("candidates", []) if c.get("sql")]

    if not candidates:
        print("  -> no valid candidates generated; logging and skipping")
        log_event({
            "type": "no_candidates",
            "query": query[:300],
            "p_slow": baseline["p_slow"],
            "predicted_ms": baseline["predicted_ms"],
            "reason": llm_result.get("reason", ""),
            "llm_source": llm_result.get("llm_source", "unknown"),
            "timestamp": datetime.datetime.now().isoformat(),
        })
        return None

    # 4. Model-in-the-loop ranking — neural net ranks the LLM's candidates
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
                    for c in llm_result.get("candidates", [])}

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
        "reason": llm_result.get("reason", ""),
        "index_suggestion": llm_result.get("index_suggestion", "NONE"),
        "llm_source": llm_result.get("llm_source", "unknown"),
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
    print("  SQL Query Monitor — fine-tuned LLM + neural ranker")
    print("=" * 60)

    print("Loading neural classifier...")
    analyzer = QueryAnalyzer()
    print(f"  loaded ({analyzer.meta['n_features']}-feature multi-task model)")

    print("Loading fine-tuned LLM optimizer...")
    optimizer = LLMOptimizer()

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
                                      analyzer, optimizer, conn, schema)
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