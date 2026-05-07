"""
Template-based query generator.

Produces additional queries via parameterized templates to enrich
training_data.csv beyond the hand-written set in all_queries.py.

Templates target three gaps in the existing dataset:
  1. More slow queries (multi-table aggregations, correlated subqueries,
     unindexed ORDER BY, etc.) to address class imbalance.
  2. Borderline queries (filters expected to land near the 100ms threshold)
     so the model learns the actual decision boundary.
  3. Realistic application-style queries (top-N, leaderboards, joined
     reports) rather than synthetic syntax patterns.
"""

import csv
import os
import random
import sys
import psycopg2

# Reuse the measurement + feature code from all_queries.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from all_queries import (
    measure_query,
    extract_features,
    SLOW_THRESHOLD_MS,
)

random.seed(42)  # reproducibility


# ──────────────────────────────────────────────────────────────────────
# TEMPLATES
# ──────────────────────────────────────────────────────────────────────

def gen_slow_queries():
    """Queries we expect to be slow — to balance the dataset."""
    queries = []

    # 1. Multi-join aggregations across all 3 tables
    agg_funcs = ["COUNT", "SUM", "AVG", "MAX", "MIN"]
    agg_cols = ["p.score", "p.view_count", "p.answer_count", "v.vote_type_id"]
    group_cols = ["u.location", "u.display_name", "p.post_type_id"]
    for agg in agg_funcs:
        for col in agg_cols:
            for grp in group_cols:
                queries.append(
                    f"SELECT {grp}, {agg}({col}) "
                    f"FROM users u JOIN posts p ON p.owner_user_id = u.id "
                    f"JOIN votes v ON v.post_id = p.id "
                    f"GROUP BY {grp} ORDER BY {agg}({col}) DESC"
                )

    # 2. Correlated subqueries
    for col in ["score", "view_count", "answer_count"]:
        queries.append(
            f"SELECT u.display_name, "
            f"(SELECT COUNT(*) FROM posts p WHERE p.owner_user_id = u.id) AS post_count, "
            f"(SELECT AVG(p.{col}) FROM posts p WHERE p.owner_user_id = u.id) AS avg_{col} "
            f"FROM users u WHERE u.reputation > 100"
        )

    # 3. ORDER BY on unindexed text columns without LIMIT
    for col in ["display_name", "location"]:
        queries.append(f"SELECT * FROM users ORDER BY LOWER({col}) ASC")

    # 4. LIKE wildcard on both sides + JOIN
    keywords = ["data", "user", "stack", "test", "admin", "ai", "ml", "neural"]
    for kw in keywords:
        queries.append(
            f"SELECT u.display_name, p.title FROM users u "
            f"JOIN posts p ON p.owner_user_id = u.id "
            f"WHERE p.title LIKE '%{kw}%' OR u.display_name LIKE '%{kw}%'"
        )

    # 5. Function calls in WHERE (prevent index use)
    for fn in ["LOWER", "UPPER"]:
        for kw in ["smith", "data", "ai", "user", "test"]:
            queries.append(
                f"SELECT * FROM users WHERE {fn}(display_name) = '{kw}'"
            )

    # 6. Big OR chains
    locations = ["United States", "Germany", "France", "India", "Japan",
                 "Brazil", "Canada", "Australia", "Spain", "Italy"]
    or_clause = " OR ".join(f"location = '{loc}'" for loc in locations[:5])
    queries.append(f"SELECT * FROM users WHERE {or_clause}")
    or_clause = " OR ".join(f"location = '{loc}'" for loc in locations)
    queries.append(f"SELECT * FROM users WHERE {or_clause}")

    # 7. NOT IN with subqueries
    queries.append(
        "SELECT * FROM users WHERE id NOT IN "
        "(SELECT owner_user_id FROM posts WHERE owner_user_id IS NOT NULL)"
    )
    queries.append(
        "SELECT * FROM posts WHERE owner_user_id NOT IN "
        "(SELECT id FROM users WHERE reputation > 1000)"
    )

    # 8. CROSS JOIN-ish patterns (Cartesian-leaning)
    queries.append(
        "SELECT u1.display_name, u2.display_name FROM users u1, users u2 "
        "WHERE u1.location = u2.location AND u1.id < u2.id LIMIT 1000"
    )

    return queries


def gen_borderline_queries():
    """Queries expected to land near the 100ms threshold — teach the boundary."""
    queries = []

    # Medium-selectivity filters (not too tight, not too loose)
    for thr in [50, 100, 200, 500, 1000]:
        queries.append(
            f"SELECT * FROM posts WHERE score > {thr} ORDER BY view_count DESC"
        )
        queries.append(
            f"SELECT * FROM users WHERE reputation > {thr} ORDER BY display_name"
        )

    # Joins with selective WHERE
    for thr in [10, 50, 100]:
        queries.append(
            f"SELECT p.title, u.display_name FROM posts p "
            f"JOIN users u ON p.owner_user_id = u.id "
            f"WHERE p.score > {thr}"
        )

    # GROUP BY with HAVING (mid-cost aggregation)
    for thr in [1, 3, 5, 10]:
        queries.append(
            f"SELECT owner_user_id, COUNT(*) AS post_count "
            f"FROM posts GROUP BY owner_user_id HAVING COUNT(*) > {thr}"
        )

    # DISTINCT on unindexed columns
    queries.append("SELECT DISTINCT location FROM users")
    queries.append("SELECT DISTINCT post_type_id, owner_user_id FROM posts")

    return queries


def gen_realistic_queries():
    """Queries that look like what a real application would run."""
    queries = []

    # Top-N reports
    for n in [10, 25, 50, 100]:
        queries.append(
            f"SELECT display_name, reputation FROM users "
            f"ORDER BY reputation DESC LIMIT {n}"
        )
        queries.append(
            f"SELECT title, score, view_count FROM posts "
            f"WHERE post_type_id = 1 ORDER BY score DESC LIMIT {n}"
        )

    # User leaderboard with post stats
    queries.append(
        "SELECT u.display_name, COUNT(p.id) AS posts, SUM(p.score) AS total_score "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY u.id, u.display_name ORDER BY total_score DESC LIMIT 50"
    )

    # Posts with their vote tallies
    queries.append(
        "SELECT p.title, p.score, COUNT(v.id) AS vote_count "
        "FROM posts p LEFT JOIN votes v ON v.post_id = p.id "
        "WHERE p.post_type_id = 1 GROUP BY p.id, p.title, p.score "
        "ORDER BY vote_count DESC LIMIT 25"
    )

    # Recent activity queries
    queries.append(
        "SELECT * FROM posts WHERE creation_date > '2020-01-01' "
        "ORDER BY creation_date DESC LIMIT 50"
    )
    queries.append(
        "SELECT * FROM votes WHERE creation_date > '2022-01-01' "
        "ORDER BY creation_date DESC LIMIT 100"
    )

    # User-by-id lookups (should be very fast — primary key)
    for uid in random.sample(range(1, 70000), 20):
        queries.append(f"SELECT * FROM users WHERE id = {uid}")

    # Post-by-id lookups
    for pid in random.sample(range(1, 25000), 20):
        queries.append(f"SELECT title, score, view_count FROM posts WHERE id = {pid}")

    # Filter combos
    for score_thr in [10, 50, 100]:
        for view_thr in [100, 500, 1000]:
            queries.append(
                f"SELECT * FROM posts WHERE score > {score_thr} "
                f"AND view_count > {view_thr} LIMIT 50"
            )

    return queries


# ──────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    slow = gen_slow_queries()
    border = gen_borderline_queries()
    real = gen_realistic_queries()

    # De-duplicate while preserving order
    seen = set()
    queries = []
    for q in slow + border + real:
        if q not in seen:
            seen.add(q)
            queries.append(q)

    print(f"Generated: {len(slow)} slow-targeted, {len(border)} borderline, "
          f"{len(real)} realistic")
    print(f"After dedup: {len(queries)} total new queries\n")

    # Connect and measure
    conn = psycopg2.connect(
        dbname="stackexchange_db",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
    conn.autocommit = True

    # Load existing CSV to know our starting query_id and to append correctly
    csv_path = "training_data.csv"
    existing_rows = []
    next_id = 1
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
            if existing_rows:
                next_id = int(existing_rows[-1]["query_id"]) + 1
        print(f"Existing CSV has {len(existing_rows)} rows. "
              f"New rows will start at query_id={next_id}\n")

    new_rows = []
    n = len(queries)
    for offset, query in enumerate(queries):
        i = offset + 1
        print(f"[{i}/{n}] running... ", end="", flush=True)
        m = measure_query(conn, query)
        if m is None:
            print("ERROR — skipped")
            continue

        cold, warm_median, warm_std, all_times, timed_out_count, plan_feats = m
        text_features = extract_features(query)
        label = "slow" if warm_median > SLOW_THRESHOLD_MS else "fast"

        row = {
            "query_id": next_id + offset,
            "query_text": query,
            "execution_time_cold_ms": round(cold, 2),
            "execution_time_warm_ms": round(warm_median, 2),
            "execution_time_warm_std_ms": round(warm_std, 2),
            "all_runs_ms": ";".join(f"{t:.2f}" for t in all_times),
            "timed_out_runs": timed_out_count,
            "label": label,
        }
        row.update(text_features)
        row.update(plan_feats)
        new_rows.append(row)
        suffix = f"  [{timed_out_count} timeout(s)]" if timed_out_count > 0 else ""
        print(f"cold={cold:.1f}ms  warm={warm_median:.1f}ms  → {label}{suffix}")

    # Append to existing CSV
    if existing_rows:
        fieldnames = list(existing_rows[0].keys())
    else:
        fieldnames = list(new_rows[0].keys()) if new_rows else []

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(new_rows)

    new_slow = sum(1 for r in new_rows if r["label"] == "slow")
    new_fast = sum(1 for r in new_rows if r["label"] == "fast")
    total = len(existing_rows) + len(new_rows)
    print(f"\n── Summary ──")
    print(f"New queries measured: {len(new_rows)}  (slow: {new_slow}, fast: {new_fast})")
    print(f"Total dataset size: {total} queries")

    conn.close()


if __name__ == "__main__":
    main()