"""
Template-based query generator — diversified version.

Key improvements over original:
  - Every template is PARAMETERIZED so each run produces different queries
  - New pattern categories: CASE WHEN, EXISTS, CTEs, multi-column GROUP BY,
    date truncation, HAVING with multiple conditions, self-joins, UNION ALL
  - Random sampling ensures no two runs produce the same queries
  - Explicit deduplication against existing training_data.csv
"""

import csv
import os
import random
import sys
import psycopg2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from all_queries import (
    measure_query,
    extract_features,
    SLOW_THRESHOLD_MS,
)

random.seed(None)  # different seed every run for genuine diversity


# ── Shared parameter pools ──────────────────────────────────────────────

SCORE_THRESHOLDS   = [0, 1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 500]
REP_THRESHOLDS     = [1, 10, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
VIEW_THRESHOLDS    = [100, 500, 1000, 2000, 5000, 10000, 50000]
ANSWER_THRESHOLDS  = [0, 1, 2, 3, 5, 10, 20]
LIMITS             = [5, 10, 20, 25, 50, 100, 200, 500]
LOCATIONS          = [
    "United States", "Germany", "France", "India", "Japan", "Brazil",
    "Canada", "Australia", "Spain", "Italy", "United Kingdom", "Netherlands",
    "Sweden", "Norway", "Poland", "Mexico", "Argentina", "Russia",
    "China", "South Korea", "Singapore", "New Zealand", "Switzerland",
]
KEYWORDS           = [
    "regression", "machine learning", "neural", "bayesian", "clustering",
    "classification", "hypothesis", "variance", "distribution", "probability",
    "linear", "logistic", "time series", "p-value", "confidence", "correlation",
    "random forest", "deep learning", "sampling", "bootstrap", "causal",
    "inference", "prediction", "model", "statistics", "data science",
]
VOTE_TYPES         = [1, 2, 3, 4, 5, 8]
POST_TYPES         = [1, 2]
YEARS              = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
MONTHS             = list(range(1, 13))
AGG_FUNCS          = ["COUNT", "SUM", "AVG", "MAX", "MIN"]
AGG_COLS_POSTS     = ["p.score", "p.view_count", "p.answer_count"]
AGG_COLS_VOTES     = ["v.vote_type_id"]
GROUP_COLS         = ["u.location", "u.display_name", "p.post_type_id"]
ORDER_DIRS         = ["ASC", "DESC"]


def pick(lst, n=1):
    """Randomly pick n items from lst without replacement where possible."""
    n = min(n, len(lst))
    return random.sample(lst, n) if n > 1 else random.choice(lst)


# ── 1. CASE WHEN (new pattern) ─────────────────────────────────────────

def gen_case_when():
    queries = []

    # Bucketing scores
    low = pick(SCORE_THRESHOLDS)
    high = pick([t for t in SCORE_THRESHOLDS if t > low])
    queries.append(
        f"SELECT p.title, p.score, "
        f"CASE WHEN p.score >= {high} THEN 'high' "
        f"     WHEN p.score >= {low} THEN 'medium' "
        f"     ELSE 'low' END AS score_tier "
        f"FROM posts p JOIN users u ON p.owner_user_id = u.id "
        f"ORDER BY p.score DESC"
    )

    # Conditional aggregation (always slow without LIMIT)
    queries.append(
        "SELECT u.display_name, "
        f"COUNT(CASE WHEN p.score > {pick(SCORE_THRESHOLDS)} THEN 1 END) AS good_posts, "
        f"COUNT(CASE WHEN p.score <= 0 THEN 1 END) AS bad_posts, "
        "COUNT(p.id) AS total_posts "
        "FROM users u LEFT JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY u.id, u.display_name "
        "ORDER BY good_posts DESC"
    )

    # Vote type breakdown per user
    queries.append(
        "SELECT u.display_name, "
        "COUNT(CASE WHEN v.vote_type_id = 1 THEN 1 END) AS upvotes, "
        "COUNT(CASE WHEN v.vote_type_id = 2 THEN 1 END) AS downvotes, "
        "COUNT(CASE WHEN v.vote_type_id = 3 THEN 1 END) AS other_votes "
        "FROM users u "
        "JOIN posts p ON p.owner_user_id = u.id "
        "JOIN votes v ON v.post_id = p.id "
        "GROUP BY u.id, u.display_name "
        "ORDER BY upvotes DESC"
    )

    # Post type conditional
    lim = pick(LIMITS)
    queries.append(
        "SELECT p.title, p.score, "
        "CASE WHEN p.post_type_id = 1 THEN 'question' "
        "     WHEN p.post_type_id = 2 THEN 'answer' "
        "     ELSE 'other' END AS post_kind "
        f"FROM posts p WHERE p.score > {pick(SCORE_THRESHOLDS)} "
        f"ORDER BY p.score DESC LIMIT {lim}"
    )

    return queries


# ── 2. EXISTS subqueries (new pattern) ────────────────────────────────

def gen_exists():
    queries = []

    rep = pick(REP_THRESHOLDS)
    queries.append(
        f"SELECT u.display_name, u.reputation FROM users u "
        f"WHERE EXISTS (SELECT 1 FROM posts p WHERE p.owner_user_id = u.id AND p.score > {pick(SCORE_THRESHOLDS)}) "
        f"AND u.reputation > {rep} "
        f"ORDER BY u.reputation DESC"
    )

    queries.append(
        f"SELECT p.title, p.score FROM posts p "
        f"WHERE EXISTS (SELECT 1 FROM votes v WHERE v.post_id = p.id AND v.vote_type_id = {pick(VOTE_TYPES)}) "
        f"AND p.score > {pick(SCORE_THRESHOLDS)} "
        f"ORDER BY p.score DESC"
    )

    queries.append(
        "SELECT u.display_name FROM users u "
        "WHERE NOT EXISTS (SELECT 1 FROM posts p WHERE p.owner_user_id = u.id) "
        f"AND u.reputation > {pick(REP_THRESHOLDS)} "
        f"ORDER BY u.reputation DESC LIMIT {pick(LIMITS)}"
    )

    queries.append(
        "SELECT p.title, p.score FROM posts p "
        "WHERE NOT EXISTS (SELECT 1 FROM votes v WHERE v.post_id = p.id) "
        f"AND p.score > {pick(SCORE_THRESHOLDS)} "
        f"ORDER BY p.score DESC LIMIT {pick(LIMITS)}"
    )

    return queries


# ── 3. CTEs / WITH clauses (new pattern) ──────────────────────────────

def gen_cte():
    queries = []

    rep = pick(REP_THRESHOLDS)
    lim = pick(LIMITS)
    queries.append(
        f"WITH top_users AS ( "
        f"  SELECT id, display_name, reputation FROM users WHERE reputation > {rep} "
        f") "
        f"SELECT tu.display_name, COUNT(p.id) AS post_count "
        f"FROM top_users tu LEFT JOIN posts p ON p.owner_user_id = tu.id "
        f"GROUP BY tu.id, tu.display_name "
        f"ORDER BY post_count DESC LIMIT {lim}"
    )

    score = pick(SCORE_THRESHOLDS)
    queries.append(
        f"WITH scored_posts AS ( "
        f"  SELECT id, title, score, owner_user_id FROM posts WHERE score > {score} "
        f") "
        f"SELECT u.display_name, COUNT(sp.id) AS high_score_posts, AVG(sp.score) AS avg_score "
        f"FROM users u JOIN scored_posts sp ON sp.owner_user_id = u.id "
        f"GROUP BY u.id, u.display_name "
        f"ORDER BY high_score_posts DESC"
    )

    vt = pick(VOTE_TYPES)
    queries.append(
        f"WITH vote_counts AS ( "
        f"  SELECT post_id, COUNT(*) AS cnt FROM votes WHERE vote_type_id = {vt} GROUP BY post_id "
        f") "
        f"SELECT p.title, p.score, vc.cnt AS vote_count "
        f"FROM posts p JOIN vote_counts vc ON vc.post_id = p.id "
        f"ORDER BY vc.cnt DESC LIMIT {pick(LIMITS)}"
    )

    return queries


# ── 4. Date/time filtering (new pattern) ──────────────────────────────

def gen_date_queries():
    queries = []

    year = pick(YEARS)
    queries.append(
        f"SELECT u.display_name, COUNT(p.id) AS posts "
        f"FROM users u JOIN posts p ON p.owner_user_id = u.id "
        f"WHERE EXTRACT(YEAR FROM p.creation_date) = {year} "
        f"GROUP BY u.id, u.display_name ORDER BY posts DESC"
    )

    month = pick(MONTHS)
    queries.append(
        f"SELECT COUNT(*) AS post_count, AVG(score) AS avg_score "
        f"FROM posts WHERE EXTRACT(MONTH FROM creation_date) = {month}"
    )

    year = pick(YEARS)
    queries.append(
        f"SELECT u.location, COUNT(p.id) AS posts "
        f"FROM users u JOIN posts p ON p.owner_user_id = u.id "
        f"WHERE EXTRACT(YEAR FROM u.creation_date) >= {year} "
        f"GROUP BY u.location ORDER BY posts DESC"
    )

    year1, year2 = sorted(random.sample(YEARS, 2))
    queries.append(
        f"SELECT p.title, p.score, u.display_name "
        f"FROM posts p JOIN users u ON p.owner_user_id = u.id "
        f"WHERE p.creation_date BETWEEN '{year1}-01-01' AND '{year2}-12-31' "
        f"ORDER BY p.score DESC LIMIT {pick(LIMITS)}"
    )

    return queries


# ── 5. Multi-column GROUP BY (extended) ───────────────────────────────

def gen_multicolumn_groupby():
    queries = []

    # year + location grouping
    queries.append(
        "SELECT EXTRACT(YEAR FROM p.creation_date) AS year, u.location, "
        "COUNT(p.id) AS posts, AVG(p.score) AS avg_score "
        "FROM posts p JOIN users u ON p.owner_user_id = u.id "
        "GROUP BY EXTRACT(YEAR FROM p.creation_date), u.location "
        "ORDER BY year DESC, posts DESC"
    )

    # post_type + year
    queries.append(
        "SELECT p.post_type_id, EXTRACT(YEAR FROM p.creation_date) AS year, "
        "COUNT(*) AS cnt, AVG(p.score) AS avg_score "
        "FROM posts p "
        "GROUP BY p.post_type_id, EXTRACT(YEAR FROM p.creation_date) "
        "ORDER BY year, p.post_type_id"
    )

    # 3-column grouping
    rep = pick(REP_THRESHOLDS)
    queries.append(
        f"SELECT u.location, p.post_type_id, "
        f"COUNT(p.id) AS posts, SUM(p.score) AS total_score "
        f"FROM users u JOIN posts p ON p.owner_user_id = u.id "
        f"WHERE u.reputation > {rep} "
        f"GROUP BY u.location, p.post_type_id "
        f"ORDER BY total_score DESC"
    )

    return queries


# ── 6. HAVING with complex conditions (extended) ──────────────────────

def gen_having():
    queries = []

    cnt1 = pick([2, 3, 5, 10, 20])
    avg1 = pick(SCORE_THRESHOLDS)
    queries.append(
        f"SELECT u.display_name, COUNT(p.id) AS posts, AVG(p.score) AS avg_score "
        f"FROM users u JOIN posts p ON p.owner_user_id = u.id "
        f"GROUP BY u.id, u.display_name "
        f"HAVING COUNT(p.id) > {cnt1} AND AVG(p.score) > {avg1} "
        f"ORDER BY avg_score DESC"
    )

    cnt2 = pick([5, 10, 20, 50])
    queries.append(
        f"SELECT u.location, COUNT(DISTINCT u.id) AS users, AVG(p.score) AS avg_score "
        f"FROM users u JOIN posts p ON p.owner_user_id = u.id "
        f"GROUP BY u.location "
        f"HAVING COUNT(DISTINCT u.id) > {cnt2} "
        f"ORDER BY avg_score DESC LIMIT {pick(LIMITS)}"
    )

    cnt3 = pick([3, 5, 10])
    queries.append(
        f"SELECT p.post_type_id, COUNT(*) AS cnt, SUM(p.score) AS total "
        f"FROM posts p "
        f"GROUP BY p.post_type_id "
        f"HAVING COUNT(*) > {cnt3} "
        f"ORDER BY total DESC"
    )

    return queries


# ── 7. Self-joins (new pattern) ────────────────────────────────────────

def gen_self_join():
    queries = []

    queries.append(
        "SELECT p1.title AS post1, p2.title AS post2, "
        "p1.score AS score1, p2.score AS score2 "
        "FROM posts p1 JOIN posts p2 ON p1.owner_user_id = p2.owner_user_id "
        f"WHERE p1.id < p2.id AND p1.score > {pick(SCORE_THRESHOLDS)} "
        f"LIMIT {pick(LIMITS)}"
    )

    queries.append(
        "SELECT u1.display_name AS user1, u2.display_name AS user2 "
        "FROM users u1 JOIN users u2 ON u1.location = u2.location "
        f"WHERE u1.id < u2.id AND u1.reputation > {pick(REP_THRESHOLDS)} "
        f"LIMIT {pick(LIMITS)}"
    )

    return queries


# ── 8. UNION ALL (extended) ───────────────────────────────────────────

def gen_union():
    queries = []

    s1, s2 = sorted(random.sample(SCORE_THRESHOLDS, 2))
    queries.append(
        f"SELECT id, title, score, 'high' AS tier FROM posts WHERE score > {s2} "
        f"UNION ALL "
        f"SELECT id, title, score, 'medium' AS tier FROM posts WHERE score BETWEEN {s1} AND {s2}"
    )

    rep1, rep2 = sorted(random.sample(REP_THRESHOLDS, 2))
    queries.append(
        f"SELECT id, display_name, reputation, 'top' AS tier FROM users WHERE reputation > {rep2} "
        f"UNION ALL "
        f"SELECT id, display_name, reputation, 'mid' AS tier FROM users WHERE reputation BETWEEN {rep1} AND {rep2}"
    )

    lim = pick(LIMITS)
    queries.append(
        f"SELECT p.id, p.title, u.display_name, 'author' AS role "
        f"FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.score > {pick(SCORE_THRESHOLDS)} "
        f"UNION "
        f"SELECT p.id, p.title, u.display_name, 'voter' AS role "
        f"FROM posts p JOIN votes v ON v.post_id = p.id JOIN users u ON u.id IS NOT NULL "
        f"WHERE v.vote_type_id = {pick(VOTE_TYPES)} LIMIT {lim}"
    )

    return queries


# ── 9. Multi-table aggregations (parameterized, avoid duplicates) ──────

def gen_slow_aggregations():
    queries = []
    agg = pick(AGG_FUNCS)
    col = pick(AGG_COLS_POSTS + AGG_COLS_VOTES)
    grp = pick(GROUP_COLS)
    # Only generate a few random combos per run to avoid the full Cartesian product
    combos = [(f, c, g)
              for f in random.sample(AGG_FUNCS, 3)
              for c in random.sample(AGG_COLS_POSTS, 2)
              for g in random.sample(GROUP_COLS, 2)]
    random.shuffle(combos)
    for f, c, g in combos[:6]:
        queries.append(
            f"SELECT {g}, {f}({c}) AS result "
            f"FROM users u JOIN posts p ON p.owner_user_id = u.id "
            f"JOIN votes v ON v.post_id = p.id "
            f"GROUP BY {g} ORDER BY result DESC LIMIT {pick(LIMITS)}"
        )
    return queries


# ── 10. Borderline queries (parameterized) ────────────────────────────

def gen_borderline():
    queries = []

    for _ in range(4):
        score = pick(SCORE_THRESHOLDS)
        lim = pick(LIMITS)
        queries.append(
            f"SELECT p.title, p.score, u.display_name, u.reputation "
            f"FROM posts p JOIN users u ON p.owner_user_id = u.id "
            f"WHERE p.score > {score} AND u.reputation > {pick(REP_THRESHOLDS)} "
            f"ORDER BY p.score DESC LIMIT {lim}"
        )

    for _ in range(3):
        rep = pick(REP_THRESHOLDS)
        lim = pick(LIMITS)
        loc = pick(LOCATIONS)
        queries.append(
            f"SELECT u.display_name, u.reputation, COUNT(p.id) AS posts "
            f"FROM users u LEFT JOIN posts p ON p.owner_user_id = u.id "
            f"WHERE u.reputation > {rep} AND u.location LIKE '%{loc[:6]}%' "
            f"GROUP BY u.id, u.display_name, u.reputation "
            f"ORDER BY posts DESC LIMIT {lim}"
        )

    for _ in range(3):
        vt = pick(VOTE_TYPES)
        score = pick(SCORE_THRESHOLDS)
        lim = pick(LIMITS)
        queries.append(
            f"SELECT p.title, COUNT(v.id) AS votes "
            f"FROM posts p JOIN votes v ON v.post_id = p.id "
            f"WHERE v.vote_type_id = {vt} AND p.score > {score} "
            f"GROUP BY p.id, p.title ORDER BY votes DESC LIMIT {lim}"
        )

    return queries


# ── 11. Realistic app queries (parameterized) ─────────────────────────

def gen_realistic():
    queries = []

    # Random user/post lookups (different every run)
    for uid in random.sample(range(1, 350000), 15):
        queries.append(f"SELECT * FROM users WHERE id = {uid}")
    for pid in random.sample(range(1, 430000), 15):
        queries.append(f"SELECT title, score, view_count FROM posts WHERE id = {pid}")

    # Top-N with random limits
    for _ in range(4):
        lim = pick(LIMITS)
        queries.append(
            f"SELECT display_name, reputation FROM users "
            f"ORDER BY reputation DESC LIMIT {lim}"
        )
        queries.append(
            f"SELECT title, score, view_count FROM posts "
            f"WHERE post_type_id = {pick(POST_TYPES)} "
            f"ORDER BY score DESC LIMIT {lim}"
        )

    # Recent posts/votes with random dates
    year = pick(YEARS)
    queries.append(
        f"SELECT * FROM posts WHERE creation_date > '{year}-01-01' "
        f"ORDER BY creation_date DESC LIMIT {pick(LIMITS)}"
    )
    year = pick(YEARS)
    queries.append(
        f"SELECT * FROM votes WHERE creation_date > '{year}-06-01' "
        f"ORDER BY creation_date DESC LIMIT {pick(LIMITS)}"
    )

    return queries


# ── 12. Guaranteed slow — hand-crafted to cover missing patterns ──────

def gen_guaranteed_slow():
    """
    Hand-crafted queries covering patterns with zero or few training pairs:
    CASE WHEN, EXISTS, CTEs, UNION, extra WINDOW functions.
    All verified to be slow on a Stats StackExchange dataset.
    """
    return [
        # ── CASE WHEN ──────────────────────────────────────────────────
        "SELECT u.display_name, "
        "COUNT(CASE WHEN p.score > 10 THEN 1 END) AS good_posts, "
        "COUNT(CASE WHEN p.score <= 0 THEN 1 END) AS bad_posts, "
        "COUNT(p.id) AS total "
        "FROM users u LEFT JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY u.id, u.display_name ORDER BY good_posts DESC",

        "SELECT u.display_name, "
        "SUM(CASE WHEN p.post_type_id = 1 THEN 1 ELSE 0 END) AS questions, "
        "SUM(CASE WHEN p.post_type_id = 2 THEN 1 ELSE 0 END) AS answers "
        "FROM users u LEFT JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY u.id, u.display_name ORDER BY questions DESC",

        "SELECT u.display_name, "
        "COUNT(CASE WHEN v.vote_type_id = 1 THEN 1 END) AS upvotes, "
        "COUNT(CASE WHEN v.vote_type_id = 2 THEN 1 END) AS downvotes "
        "FROM users u "
        "JOIN posts p ON p.owner_user_id = u.id "
        "JOIN votes v ON v.post_id = p.id "
        "GROUP BY u.id, u.display_name ORDER BY upvotes DESC",

        "SELECT p.title, p.score, "
        "CASE WHEN p.score >= 100 THEN 'viral' "
        "     WHEN p.score >= 10 THEN 'popular' "
        "     WHEN p.score > 0 THEN 'normal' "
        "     ELSE 'poor' END AS tier, "
        "u.display_name "
        "FROM posts p JOIN users u ON p.owner_user_id = u.id "
        "ORDER BY p.score DESC",

        "SELECT u.location, "
        "AVG(CASE WHEN p.post_type_id = 1 THEN p.score END) AS avg_question_score, "
        "AVG(CASE WHEN p.post_type_id = 2 THEN p.score END) AS avg_answer_score "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY u.location ORDER BY avg_question_score DESC",

        "SELECT u.display_name, "
        "SUM(CASE WHEN p.score > 0 THEN p.score ELSE 0 END) AS positive_score, "
        "SUM(CASE WHEN p.score < 0 THEN p.score ELSE 0 END) AS negative_score "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY u.id, u.display_name ORDER BY positive_score DESC",

        # ── EXISTS ─────────────────────────────────────────────────────
        "SELECT u.display_name, u.reputation "
        "FROM users u "
        "WHERE EXISTS (SELECT 1 FROM posts p WHERE p.owner_user_id = u.id AND p.score > 50) "
        "ORDER BY u.reputation DESC",

        "SELECT u.display_name, u.reputation "
        "FROM users u "
        "WHERE EXISTS (SELECT 1 FROM posts p WHERE p.owner_user_id = u.id "
        "  AND EXISTS (SELECT 1 FROM votes v WHERE v.post_id = p.id AND v.vote_type_id = 2)) "
        "ORDER BY u.reputation DESC",

        "SELECT p.title, p.score "
        "FROM posts p "
        "WHERE EXISTS (SELECT 1 FROM votes v WHERE v.post_id = p.id AND v.vote_type_id = 2) "
        "AND p.score > 5 "
        "ORDER BY p.score DESC",

        "SELECT u.display_name FROM users u "
        "WHERE NOT EXISTS (SELECT 1 FROM posts p WHERE p.owner_user_id = u.id) "
        "ORDER BY u.reputation DESC",

        "SELECT u.display_name, u.reputation "
        "FROM users u "
        "WHERE EXISTS (SELECT 1 FROM posts p WHERE p.owner_user_id = u.id AND p.answer_count > 5) "
        "AND u.reputation > 100 "
        "ORDER BY u.reputation DESC",

        "SELECT p.title, p.score FROM posts p "
        "WHERE NOT EXISTS (SELECT 1 FROM votes v WHERE v.post_id = p.id) "
        "AND p.score > 0 ORDER BY p.score DESC",

        # ── CTEs ───────────────────────────────────────────────────────
        "WITH active_users AS ( "
        "  SELECT id, display_name, reputation "
        "  FROM users WHERE reputation > 500 "
        ") "
        "SELECT au.display_name, COUNT(p.id) AS posts, AVG(p.score) AS avg_score "
        "FROM active_users au LEFT JOIN posts p ON p.owner_user_id = au.id "
        "GROUP BY au.id, au.display_name ORDER BY posts DESC",

        "WITH post_stats AS ( "
        "  SELECT owner_user_id, COUNT(*) AS cnt, AVG(score) AS avg_score, SUM(view_count) AS total_views "
        "  FROM posts GROUP BY owner_user_id "
        ") "
        "SELECT u.display_name, ps.cnt, ps.avg_score, ps.total_views "
        "FROM users u JOIN post_stats ps ON ps.owner_user_id = u.id "
        "ORDER BY ps.total_views DESC",

        "WITH vote_summary AS ( "
        "  SELECT p.owner_user_id, COUNT(v.id) AS total_votes "
        "  FROM votes v JOIN posts p ON v.post_id = p.id "
        "  GROUP BY p.owner_user_id "
        ") "
        "SELECT u.display_name, u.reputation, vs.total_votes "
        "FROM users u JOIN vote_summary vs ON vs.owner_user_id = u.id "
        "ORDER BY vs.total_votes DESC",

        "WITH top_posts AS ( "
        "  SELECT id, title, score, owner_user_id "
        "  FROM posts WHERE score > 20 "
        "), "
        "post_votes AS ( "
        "  SELECT post_id, COUNT(*) AS vote_count FROM votes GROUP BY post_id "
        ") "
        "SELECT tp.title, tp.score, COALESCE(pv.vote_count, 0) AS votes, u.display_name "
        "FROM top_posts tp "
        "JOIN users u ON tp.owner_user_id = u.id "
        "LEFT JOIN post_votes pv ON pv.post_id = tp.id "
        "ORDER BY votes DESC",

        "WITH user_locations AS ( "
        "  SELECT location, COUNT(*) AS user_count, AVG(reputation) AS avg_rep "
        "  FROM users WHERE location IS NOT NULL AND location != '' "
        "  GROUP BY location "
        ") "
        "SELECT ul.location, ul.user_count, ul.avg_rep, COUNT(p.id) AS total_posts "
        "FROM user_locations ul "
        "JOIN users u ON u.location = ul.location "
        "JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY ul.location, ul.user_count, ul.avg_rep "
        "ORDER BY total_posts DESC",

        "WITH ranked_users AS ( "
        "  SELECT id, display_name, reputation, "
        "  RANK() OVER (ORDER BY reputation DESC) AS rep_rank "
        "  FROM users "
        ") "
        "SELECT ru.display_name, ru.reputation, ru.rep_rank, COUNT(p.id) AS posts "
        "FROM ranked_users ru LEFT JOIN posts p ON p.owner_user_id = ru.id "
        "WHERE ru.rep_rank <= 1000 "
        "GROUP BY ru.id, ru.display_name, ru.reputation, ru.rep_rank "
        "ORDER BY ru.rep_rank",

        # ── UNION (varied) ─────────────────────────────────────────────
        "SELECT u.id, u.display_name, u.reputation, 'high_rep' AS category "
        "FROM users u WHERE u.reputation > 1000 "
        "UNION ALL "
        "SELECT u.id, u.display_name, u.reputation, 'low_rep' AS category "
        "FROM users u WHERE u.reputation < 10 "
        "ORDER BY reputation DESC",

        "SELECT p.id, p.title, p.score, 'high_score' AS category "
        "FROM posts p WHERE p.score > 50 "
        "UNION ALL "
        "SELECT p.id, p.title, p.score, 'high_views' AS category "
        "FROM posts p WHERE p.view_count > 10000 "
        "ORDER BY score DESC",

        "SELECT u.display_name, p.title, p.score, 'asked' AS involvement "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE p.post_type_id = 1 "
        "UNION "
        "SELECT u.display_name, p.title, p.score, 'answered' AS involvement "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE p.post_type_id = 2 "
        "ORDER BY score DESC",

        "SELECT 'posts' AS source, COUNT(*) AS total, AVG(score) AS avg_score FROM posts "
        "UNION ALL "
        "SELECT 'questions', COUNT(*), AVG(score) FROM posts WHERE post_type_id = 1 "
        "UNION ALL "
        "SELECT 'answers', COUNT(*), AVG(score) FROM posts WHERE post_type_id = 2",

        "SELECT u.location, COUNT(*) AS count, 'users' AS type "
        "FROM users u WHERE u.location IS NOT NULL "
        "GROUP BY u.location "
        "UNION ALL "
        "SELECT u.location, COUNT(p.id), 'posts' AS type "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id "
        "WHERE u.location IS NOT NULL "
        "GROUP BY u.location "
        "ORDER BY count DESC",

        # ── WINDOW functions (more variety) ────────────────────────────
        "SELECT u.display_name, p.score, "
        "SUM(p.score) OVER (PARTITION BY u.id ORDER BY p.creation_date) AS running_score "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id "
        "ORDER BY u.id, p.creation_date",

        "SELECT p.title, p.score, p.post_type_id, "
        "AVG(p.score) OVER (PARTITION BY p.post_type_id) AS type_avg, "
        "p.score - AVG(p.score) OVER (PARTITION BY p.post_type_id) AS diff_from_avg "
        "FROM posts p ORDER BY diff_from_avg DESC",

        "SELECT u.display_name, u.reputation, "
        "NTILE(4) OVER (ORDER BY u.reputation DESC) AS quartile "
        "FROM users u ORDER BY u.reputation DESC",

        "SELECT p.title, p.score, "
        "LAG(p.score) OVER (PARTITION BY p.post_type_id ORDER BY p.creation_date) AS prev_score, "
        "p.score - LAG(p.score) OVER (PARTITION BY p.post_type_id ORDER BY p.creation_date) AS delta "
        "FROM posts p ORDER BY p.creation_date",

        "SELECT u.display_name, p.score, "
        "ROW_NUMBER() OVER (PARTITION BY u.id ORDER BY p.score DESC) AS score_rank "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id "
        "ORDER BY u.id, score_rank",

        "SELECT p.title, p.score, "
        "FIRST_VALUE(p.title) OVER (PARTITION BY p.post_type_id ORDER BY p.score DESC) AS top_post "
        "FROM posts p",

        # ── Date/time aggregations ─────────────────────────────────────
        "SELECT EXTRACT(YEAR FROM p.creation_date) AS year, "
        "EXTRACT(MONTH FROM p.creation_date) AS month, "
        "COUNT(*) AS posts, AVG(p.score) AS avg_score "
        "FROM posts p "
        "GROUP BY year, month ORDER BY year DESC, month DESC",

        "SELECT EXTRACT(YEAR FROM u.creation_date) AS join_year, "
        "COUNT(*) AS new_users, AVG(u.reputation) AS avg_rep "
        "FROM users u GROUP BY join_year ORDER BY join_year DESC",

        "SELECT u.display_name, "
        "EXTRACT(YEAR FROM MIN(p.creation_date)) AS first_post_year, "
        "EXTRACT(YEAR FROM MAX(p.creation_date)) AS last_post_year, "
        "COUNT(p.id) AS total_posts "
        "FROM users u JOIN posts p ON p.owner_user_id = u.id "
        "GROUP BY u.id, u.display_name ORDER BY total_posts DESC",

        # ── Complex multi-join with filters ───────────────────────────
        "SELECT u.display_name, u.location, "
        "COUNT(DISTINCT p.id) AS posts, "
        "COUNT(DISTINCT v.id) AS votes_received, "
        "AVG(p.score) AS avg_score "
        "FROM users u "
        "JOIN posts p ON p.owner_user_id = u.id "
        "LEFT JOIN votes v ON v.post_id = p.id "
        "WHERE u.reputation > 100 AND p.score > 0 "
        "GROUP BY u.id, u.display_name, u.location "
        "ORDER BY votes_received DESC",

        "SELECT p.post_type_id, u.location, "
        "COUNT(p.id) AS posts, "
        "AVG(p.score) AS avg_score, "
        "SUM(p.view_count) AS total_views "
        "FROM posts p "
        "JOIN users u ON p.owner_user_id = u.id "
        "JOIN votes v ON v.post_id = p.id "
        "WHERE u.location IS NOT NULL "
        "GROUP BY p.post_type_id, u.location "
        "ORDER BY total_views DESC",
    ]


# ── Main ──────────────────────────────────────────────────────────────

def main():
    all_generated = (
        gen_guaranteed_slow() +
        gen_case_when() +
        gen_exists() +
        gen_cte() +
        gen_date_queries() +
        gen_multicolumn_groupby() +
        gen_having() +
        gen_self_join() +
        gen_union() +
        gen_slow_aggregations() +
        gen_borderline() +
        gen_realistic()
    )

    # Load existing queries to avoid duplicates
    csv_path = "training_data.csv"
    existing_rows = []
    existing_queries = set()
    next_id = 1
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                existing_queries.add(row["query_text"].strip())
        if existing_rows:
            next_id = int(existing_rows[-1]["query_id"]) + 1

    # Deduplicate against existing AND within new batch
    seen = set(existing_queries)
    queries = []
    for q in all_generated:
        q_stripped = q.strip()
        if q_stripped not in seen:
            seen.add(q_stripped)
            queries.append(q)

    print(f"Existing CSV: {len(existing_rows)} rows")
    print(f"Generated: {len(all_generated)} candidates")
    print(f"New (after dedup): {len(queries)} queries to measure\n")

    if not queries:
        print("No new queries — try running again (different random seed each time)")
        return

    conn = psycopg2.connect(
        dbname="stackexchange_db",
        user="postgres",
        password="postgres",
        host="localhost",
        port="5432"
    )
    conn.autocommit = True

    new_rows = []
    for i, query in enumerate(queries, 1):
        print(f"[{i}/{len(queries)}] running... ", end="", flush=True)
        m = measure_query(conn, query)
        if m is None:
            print("ERROR — skipped")
            continue

        cold, warm_median, warm_std, all_times, timed_out_count, plan_feats = m
        text_features = extract_features(query)
        label = "slow" if warm_median > SLOW_THRESHOLD_MS else "fast"

        row = {
            "query_id": next_id + len(new_rows),
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

    if not new_rows:
        print("No rows measured.")
        conn.close()
        return

    fieldnames = list(existing_rows[0].keys()) if existing_rows else list(new_rows[0].keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writerows(new_rows)

    new_slow = sum(1 for r in new_rows if r["label"] == "slow")
    new_fast = sum(1 for r in new_rows if r["label"] == "fast")
    print(f"\n── Summary ──")
    print(f"New queries measured: {len(new_rows)}  (slow: {new_slow}, fast: {new_fast})")
    print(f"Total dataset size: {len(existing_rows) + len(new_rows)} queries")

    conn.close()


if __name__ == "__main__":
    main()