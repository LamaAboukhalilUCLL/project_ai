import psycopg2
import psycopg2.errors
import csv
import re
import time
import statistics

"""
For each query, run it 6 times back-to-back with DISCARD ALL before the first run:

- Run 1 = "cold-ish" timing. PostgreSQL's session-level state is cleared. Buffer pool isn't, but it's the most representative first-run number we can practically get.
- Runs 2–6 = warm cache. Take the median (not mean — median ignores the occasional weird outlier).
- We save both numbers as separate columns (execution_time_cold_ms, execution_time_warm_ms) plus the standard deviation of the warm runs (so we can show stability) and the raw list of all 6 runs (for transparency in the report).

"""

RUNS_PER_QUERY = 6  # 1 first-run + 5 warm-cache runs (reduced for slow queries)
SLOW_THRESHOLD_MS = 100
TIMEOUT_MS = 10000  # 10s statement timeout cap


def run_explain_analyze(cur, query):
    """
    Run EXPLAIN ANALYZE and return (execution_time_ms, timed_out_bool).
    A timeout returns (TIMEOUT_MS, True) so we still capture the query
    as a (very) slow data point instead of throwing it away.
    """
    try:
        cur.execute(f"SET statement_timeout = '{TIMEOUT_MS}ms'")
        cur.execute(f"EXPLAIN ANALYZE {query}")
        output = cur.fetchall()
        for row in output:
            if "Execution Time:" in row[0]:
                m = re.search(r"Execution Time: ([\d.]+)", row[0])
                if m:
                    return float(m.group(1)), False
        return None, False
    except psycopg2.errors.QueryCanceled:
        return float(TIMEOUT_MS), True


def measure_query(conn, query):
    """
    Run the query several times and return cold + warm timings.
    Adapts the number of warm runs based on how slow the cold run was —
    no point running a 10-second query 5 more times.
    """
    cur = conn.cursor()
    times = []
    timed_out_count = 0
    try:
        cur.execute("DISCARD ALL")

        # Cold run
        t, to = run_explain_analyze(cur, query)
        if t is None:
            return None  # genuine error, not a timeout
        times.append(t)
        if to:
            timed_out_count += 1

        # Adapt warm-run count to cold time
        if to or t > 2000:
            warm_runs = 1   # very slow — one confirmation is enough
        elif t > 500:
            warm_runs = 2
        else:
            warm_runs = 5   # fast — get a stable median

        for _ in range(warm_runs):
            t, to = run_explain_analyze(cur, query)
            if t is None:
                break
            times.append(t)
            if to:
                timed_out_count += 1
            time.sleep(0.05)
    except Exception as e:
        print(f"[ERROR: {e}]", end=" ", flush=True)
        return None
    finally:
        cur.close()

    if not times:
        return None

    cold_time = times[0]
    warm_times = times[1:] if len(times) > 1 else [times[0]]
    warm_median = statistics.median(warm_times)
    warm_std = statistics.pstdev(warm_times) if len(warm_times) > 1 else 0.0
    return cold_time, warm_median, warm_std, times, timed_out_count

conn = psycopg2.connect(
    dbname="stackexchange_db",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)
conn.autocommit = True
cur = conn.cursor()

queries = [
    # ══════════════════════════════════════
    # SLOW QUERIES
    # ══════════════════════════════════════

    # ── SELECT * no filter ──
    "SELECT * FROM posts",
    "SELECT * FROM users",
    "SELECT * FROM votes",
    "SELECT * FROM posts ORDER BY score DESC",
    "SELECT * FROM users ORDER BY reputation DESC",
    "SELECT * FROM votes ORDER BY creation_date ASC",
    "SELECT * FROM posts ORDER BY view_count DESC",
    "SELECT * FROM users ORDER BY creation_date ASC",
    "SELECT * FROM posts ORDER BY answer_count DESC",
    "SELECT * FROM votes ORDER BY vote_type_id ASC",

    # ── SELECT * with JOIN ──
    "SELECT * FROM posts p JOIN users u ON p.owner_user_id = u.id",
    "SELECT * FROM posts p JOIN votes v ON v.post_id = p.id",
    "SELECT * FROM posts p JOIN users u ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id",
    "SELECT * FROM posts p JOIN users u ON p.owner_user_id = u.id ORDER BY p.score DESC",
    "SELECT * FROM posts p JOIN votes v ON v.post_id = p.id ORDER BY p.score DESC",
    "SELECT * FROM posts p JOIN users u ON p.owner_user_id = u.id ORDER BY u.reputation DESC",
    "SELECT * FROM posts p JOIN users u ON p.owner_user_id = u.id ORDER BY p.view_count DESC",
    "SELECT * FROM posts p JOIN users u ON p.owner_user_id = u.id ORDER BY p.creation_date DESC",
    "SELECT * FROM votes v JOIN posts p ON v.post_id = p.id ORDER BY p.score DESC",
    "SELECT * FROM votes v JOIN posts p ON v.post_id = p.id JOIN users u ON p.owner_user_id = u.id",

    # ── LIKE wildcard — users location ──
    "SELECT * FROM users WHERE location LIKE '%United States%'",
    "SELECT * FROM users WHERE location LIKE '%Germany%'",
    "SELECT * FROM users WHERE location LIKE '%United Kingdom%'",
    "SELECT * FROM users WHERE location LIKE '%Canada%'",
    "SELECT * FROM users WHERE location LIKE '%France%'",
    "SELECT * FROM users WHERE location LIKE '%Australia%'",
    "SELECT * FROM users WHERE location LIKE '%India%'",
    "SELECT * FROM users WHERE location LIKE '%Netherlands%'",
    "SELECT * FROM users WHERE location LIKE '%Brazil%'",
    "SELECT * FROM users WHERE location LIKE '%Spain%'",
    "SELECT * FROM users WHERE location LIKE '%Italy%'",
    "SELECT * FROM users WHERE location LIKE '%Sweden%'",
    "SELECT * FROM users WHERE location LIKE '%Norway%'",
    "SELECT * FROM users WHERE location LIKE '%Japan%'",
    "SELECT * FROM users WHERE location LIKE '%China%'",
    "SELECT * FROM users WHERE location LIKE '%New York%'",
    "SELECT * FROM users WHERE location LIKE '%California%'",
    "SELECT * FROM users WHERE location LIKE '%London%'",
    "SELECT * FROM users WHERE location LIKE '%Berlin%'",
    "SELECT * FROM users WHERE location LIKE '%Paris%'",
    "SELECT * FROM users WHERE location LIKE '%Toronto%'",
    "SELECT * FROM users WHERE location LIKE '%Amsterdam%'",
    "SELECT * FROM users WHERE location LIKE '%Singapore%'",
    "SELECT * FROM users WHERE location LIKE '%Moscow%'",
    "SELECT * FROM users WHERE location LIKE '%Sydney%'",

    # ── LIKE wildcard — display_name ──
    "SELECT * FROM users WHERE display_name LIKE '%john%'",
    "SELECT * FROM users WHERE display_name LIKE '%smith%'",
    "SELECT * FROM users WHERE display_name LIKE '%data%'",
    "SELECT * FROM users WHERE display_name LIKE '%stat%'",
    "SELECT * FROM users WHERE display_name LIKE '%user%'",
    "SELECT * FROM users WHERE display_name LIKE '%alex%'",
    "SELECT * FROM users WHERE display_name LIKE '%mike%'",
    "SELECT * FROM users WHERE display_name LIKE '%chris%'",
    "SELECT * FROM users WHERE display_name LIKE '%admin%'",
    "SELECT * FROM users WHERE display_name LIKE '%dev%'",
    "SELECT * FROM users WHERE display_name LIKE '%stats%'",
    "SELECT * FROM users WHERE display_name LIKE '%math%'",
    "SELECT * FROM users WHERE display_name LIKE '%professor%'",
    "SELECT * FROM users WHERE display_name LIKE '%doctor%'",
    "SELECT * FROM users WHERE display_name LIKE '%analyst%'",

    # ── LIKE wildcard — posts title ──
    "SELECT * FROM posts WHERE title LIKE '%regression%'",
    "SELECT * FROM posts WHERE title LIKE '%machine learning%'",
    "SELECT * FROM posts WHERE title LIKE '%neural%'",
    "SELECT * FROM posts WHERE title LIKE '%bayesian%'",
    "SELECT * FROM posts WHERE title LIKE '%clustering%'",
    "SELECT * FROM posts WHERE title LIKE '%classification%'",
    "SELECT * FROM posts WHERE title LIKE '%hypothesis%'",
    "SELECT * FROM posts WHERE title LIKE '%variance%'",
    "SELECT * FROM posts WHERE title LIKE '%distribution%'",
    "SELECT * FROM posts WHERE title LIKE '%probability%'",
    "SELECT * FROM posts WHERE title LIKE '%linear%'",
    "SELECT * FROM posts WHERE title LIKE '%logistic%'",
    "SELECT * FROM posts WHERE title LIKE '%time series%'",
    "SELECT * FROM posts WHERE title LIKE '%p-value%'",
    "SELECT * FROM posts WHERE title LIKE '%confidence%'",
    "SELECT * FROM posts WHERE title LIKE '%correlation%'",
    "SELECT * FROM posts WHERE title LIKE '%random forest%'",
    "SELECT * FROM posts WHERE title LIKE '%deep learning%'",
    "SELECT * FROM posts WHERE title LIKE '%sampling%'",
    "SELECT * FROM posts WHERE title LIKE '%bootstrap%'",

    # ── LIKE wildcard with JOIN ──
    "SELECT p.title, p.score, u.display_name, u.location FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%United States%' ORDER BY p.score DESC",
    "SELECT p.title, p.score, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%Germany%' ORDER BY p.score DESC",
    "SELECT p.title, p.score, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%United Kingdom%'",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.display_name LIKE '%john%'",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.title LIKE '%regression%'",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%Canada%'",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.title LIKE '%bayesian%'",
    "SELECT p.title, u.location FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%Australia%'",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.display_name LIKE '%data%'",
    "SELECT p.title, p.score FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.title LIKE '%neural%' ORDER BY p.score DESC",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%New York%' ORDER BY p.score DESC",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%London%' ORDER BY p.score DESC",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%California%'",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.title LIKE '%linear%' ORDER BY p.score DESC",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.title LIKE '%logistic%'",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.title LIKE '%time series%'",
    "SELECT p.title, u.location FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%Berlin%'",
    "SELECT p.title, u.location FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%Toronto%'",
    "SELECT p.title, p.score FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.display_name LIKE '%stats%'",
    "SELECT p.title, p.score FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.display_name LIKE '%math%'",

    # ── SELECT * no useful filter ──
    "SELECT * FROM posts WHERE score > 0",
    "SELECT * FROM posts WHERE score >= 0",
    "SELECT * FROM posts WHERE view_count >= 0",
    "SELECT * FROM users WHERE reputation >= 0",
    "SELECT * FROM votes WHERE vote_type_id >= 0",
    "SELECT * FROM posts WHERE answer_count >= 0",
    "SELECT * FROM posts WHERE post_type_id >= 1",
    "SELECT * FROM users WHERE id > 0",
    "SELECT * FROM posts WHERE id > 0",
    "SELECT * FROM votes WHERE id > 0",
    "SELECT * FROM posts WHERE view_count > 1000",
    "SELECT * FROM posts WHERE answer_count > 5",
    "SELECT * FROM posts WHERE post_type_id = 1",
    "SELECT * FROM users WHERE reputation > 1000",
    "SELECT * FROM users WHERE reputation BETWEEN 500 AND 2000",
    "SELECT * FROM posts WHERE score BETWEEN 10 AND 100",
    "SELECT * FROM posts WHERE creation_date > '2020-01-01'",
    "SELECT * FROM users WHERE creation_date > '2018-01-01'",
    "SELECT * FROM posts WHERE creation_date BETWEEN '2019-01-01' AND '2021-01-01'",
    "SELECT * FROM votes WHERE creation_date > '2020-01-01'",

    # ── SELECT * ORDER BY no LIMIT ──
    "SELECT * FROM posts WHERE score > 10 ORDER BY score DESC",
    "SELECT * FROM posts WHERE score > 5 ORDER BY view_count DESC",
    "SELECT * FROM users WHERE reputation > 100 ORDER BY reputation DESC",
    "SELECT * FROM users WHERE reputation > 0 ORDER BY creation_date DESC",
    "SELECT * FROM posts WHERE answer_count > 0 ORDER BY answer_count DESC",
    "SELECT * FROM posts WHERE post_type_id = 1 ORDER BY score DESC",
    "SELECT * FROM votes WHERE vote_type_id = 2 ORDER BY creation_date DESC",
    "SELECT * FROM posts WHERE view_count > 100 ORDER BY view_count DESC",
    "SELECT * FROM users WHERE reputation > 500 ORDER BY reputation ASC",
    "SELECT * FROM posts WHERE score > 20 ORDER BY creation_date DESC",

    # ── simple subqueries ──
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 1000)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 500)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 2000)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 5000)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE location LIKE '%United States%')",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE location LIKE '%Germany%')",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE score > 50)",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE score > 100)",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE answer_count > 5)",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE view_count > 1000)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 3000)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 4000)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation BETWEEN 1000 AND 5000)",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE score > 200)",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE view_count > 5000)",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE answer_count > 3)",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE creation_date > '2015-01-01')",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE creation_date > '2018-01-01')",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE post_type_id = 1 AND score > 10)",
    "SELECT * FROM posts WHERE score > (SELECT AVG(score) FROM posts WHERE post_type_id = 1)",

    # ── nested subqueries ──
    "SELECT * FROM posts WHERE id IN (SELECT post_id FROM votes WHERE post_id IN (SELECT id FROM posts WHERE score > 10))",
    "SELECT * FROM posts WHERE id IN (SELECT post_id FROM votes WHERE post_id IN (SELECT id FROM posts WHERE score > 50))",
    "SELECT * FROM posts WHERE id IN (SELECT post_id FROM votes WHERE post_id IN (SELECT id FROM posts WHERE score > 100))",
    "SELECT * FROM users WHERE id IN (SELECT owner_user_id FROM posts WHERE score > (SELECT AVG(score) FROM posts))",
    "SELECT * FROM users WHERE id IN (SELECT owner_user_id FROM posts WHERE view_count > (SELECT AVG(view_count) FROM posts))",
    "SELECT * FROM users WHERE id IN (SELECT owner_user_id FROM posts WHERE id IN (SELECT post_id FROM votes WHERE vote_type_id = 2))",
    "SELECT * FROM users WHERE id IN (SELECT owner_user_id FROM posts WHERE id IN (SELECT post_id FROM votes WHERE vote_type_id = 1))",
    "SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > (SELECT AVG(reputation) FROM users))",
    "SELECT * FROM votes WHERE post_id IN (SELECT id FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 1000))",
    "SELECT * FROM posts WHERE score > (SELECT AVG(score) FROM posts) AND view_count > (SELECT AVG(view_count) FROM posts)",

    # ── correlated subqueries ──
    "SELECT display_name, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id) as post_count FROM users",
    "SELECT display_name, (SELECT AVG(score) FROM posts WHERE owner_user_id = users.id) as avg_score FROM users",
    "SELECT display_name, (SELECT SUM(view_count) FROM posts WHERE owner_user_id = users.id) as total_views FROM users",
    "SELECT display_name, (SELECT MAX(score) FROM posts WHERE owner_user_id = users.id) as max_score FROM users",
    "SELECT display_name, (SELECT MIN(score) FROM posts WHERE owner_user_id = users.id) as min_score FROM users",
    "SELECT display_name, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id) as post_count FROM users WHERE reputation > 500",
    "SELECT display_name, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id) as post_count FROM users WHERE reputation > 1000",
    "SELECT display_name, (SELECT COUNT(*) FROM votes v JOIN posts p ON v.post_id = p.id WHERE p.owner_user_id = users.id) as vote_count FROM users",
    "SELECT display_name, (SELECT AVG(score) FROM posts WHERE owner_user_id = users.id) as avg_score, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id) as post_count FROM users",
    "SELECT display_name, (SELECT MAX(view_count) FROM posts WHERE owner_user_id = users.id) as max_views FROM users WHERE reputation > 200",
    "SELECT display_name, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id AND score > 0) as good_posts FROM users",
    "SELECT display_name, (SELECT MAX(view_count) FROM posts WHERE owner_user_id = users.id) as max_views FROM users",
    "SELECT display_name, (SELECT MIN(score) FROM posts WHERE owner_user_id = users.id) as min_score FROM users WHERE reputation > 100",
    "SELECT display_name, (SELECT COUNT(*) FROM votes v JOIN posts p ON v.post_id = p.id WHERE p.owner_user_id = users.id AND v.vote_type_id = 2) as upvotes FROM users",
    "SELECT display_name, (SELECT AVG(view_count) FROM posts WHERE owner_user_id = users.id) as avg_views FROM users WHERE reputation > 200",
    "SELECT display_name, (SELECT SUM(score) FROM posts WHERE owner_user_id = users.id) as total_score FROM users WHERE reputation > 300",
    "SELECT display_name, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id AND answer_count > 0) as answered FROM users",
    "SELECT display_name, (SELECT MAX(score) FROM posts WHERE owner_user_id = users.id) as best_score, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id) as total FROM users WHERE reputation > 100",
    "SELECT display_name, (SELECT AVG(score) FROM posts WHERE owner_user_id = users.id) as avg_score FROM users WHERE id IN (SELECT owner_user_id FROM posts WHERE view_count > 1000)",
    "SELECT display_name, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id) as posts FROM users ORDER BY posts DESC",

    # ── full table aggregations ──
    "SELECT COUNT(*), AVG(score), MAX(score), MIN(score) FROM posts",
    "SELECT COUNT(*), AVG(reputation), MAX(reputation), MIN(reputation) FROM users",
    "SELECT COUNT(*), AVG(view_count), MAX(view_count) FROM posts",
    "SELECT COUNT(*), AVG(answer_count), MAX(answer_count) FROM posts",
    "SELECT u.location, COUNT(*) as user_count, AVG(u.reputation) FROM users u GROUP BY u.location ORDER BY user_count DESC",
    "SELECT u.display_name, SUM(p.view_count) as total_views FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.display_name ORDER BY total_views DESC",
    "SELECT u.display_name, COUNT(p.id) as post_count, AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.display_name ORDER BY post_count DESC",
    "SELECT post_type_id, COUNT(*), AVG(score), MAX(score) FROM posts GROUP BY post_type_id ORDER BY COUNT(*) DESC",
    "SELECT vote_type_id, COUNT(*) FROM votes GROUP BY vote_type_id ORDER BY COUNT(*) DESC",
    "SELECT EXTRACT(YEAR FROM creation_date), COUNT(*) FROM posts GROUP BY EXTRACT(YEAR FROM creation_date) ORDER BY 1",

    # ── heavy multi-join aggregations ──
    "SELECT u.display_name, u.location, COUNT(p.id) as post_count, AVG(p.score) as avg_score, SUM(p.view_count) as total_views FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.location LIKE '%United States%' AND p.score > 0 GROUP BY u.display_name, u.location ORDER BY avg_score DESC",
    "SELECT u.display_name, COUNT(v.id) as vote_count FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY u.display_name ORDER BY vote_count DESC",
    "SELECT u.location, COUNT(p.id), AVG(p.score), SUM(p.view_count) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.location ORDER BY AVG(p.score) DESC",
    "SELECT u.display_name, COUNT(DISTINCT v.vote_type_id), SUM(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY u.display_name ORDER BY SUM(p.score) DESC",
    "SELECT u.display_name, u.location, COUNT(p.id), AVG(p.score), SUM(p.view_count), MAX(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.display_name, u.location ORDER BY SUM(p.view_count) DESC",
    "SELECT u.display_name, COUNT(p.id), AVG(p.score), SUM(p.view_count) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.location LIKE '%Germany%' GROUP BY u.display_name ORDER BY AVG(p.score) DESC",
    "SELECT u.display_name, COUNT(p.id), AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.location LIKE '%United Kingdom%' GROUP BY u.display_name ORDER BY COUNT(p.id) DESC",
    "SELECT u.display_name, SUM(p.score), COUNT(v.id) FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY u.display_name ORDER BY SUM(p.score) DESC",
    "SELECT u.location, AVG(p.score), COUNT(p.id), SUM(p.view_count) FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY u.location ORDER BY AVG(p.score) DESC",
    "SELECT u.display_name, COUNT(DISTINCT p.id), COUNT(DISTINCT v.id), AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY u.display_name ORDER BY COUNT(DISTINCT p.id) DESC",
    "SELECT u.display_name, COUNT(p.id), AVG(p.score), MAX(p.score), MIN(p.score), SUM(p.view_count) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.display_name ORDER BY AVG(p.score) DESC",
    "SELECT u.location, COUNT(DISTINCT u.id), COUNT(p.id), AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.location ORDER BY COUNT(p.id) DESC",
    "SELECT u.display_name, COUNT(p.id), SUM(p.view_count), AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE p.post_type_id = 1 GROUP BY u.display_name ORDER BY SUM(p.view_count) DESC",
    "SELECT u.display_name, COUNT(v.id) FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id WHERE v.vote_type_id = 2 GROUP BY u.display_name ORDER BY COUNT(v.id) DESC",
    "SELECT u.location, AVG(p.score), COUNT(p.id) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.location LIKE '%United States%' GROUP BY u.location",
    "SELECT EXTRACT(YEAR FROM p.creation_date), COUNT(p.id), AVG(p.score) FROM posts p JOIN users u ON p.owner_user_id = u.id GROUP BY EXTRACT(YEAR FROM p.creation_date) ORDER BY 1",
    "SELECT EXTRACT(MONTH FROM creation_date), COUNT(*), AVG(score) FROM posts GROUP BY EXTRACT(MONTH FROM creation_date) ORDER BY 1",
    "SELECT u.display_name, COUNT(DISTINCT p.post_type_id), AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.display_name HAVING COUNT(p.id) > 5 ORDER BY AVG(p.score) DESC",
    "SELECT u.location, COUNT(p.id), AVG(p.score), SUM(p.view_count), MAX(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE p.score > 0 GROUP BY u.location ORDER BY SUM(p.view_count) DESC",
    "SELECT p.post_type_id, u.location, COUNT(*), AVG(p.score) FROM posts p JOIN users u ON p.owner_user_id = u.id GROUP BY p.post_type_id, u.location ORDER BY COUNT(*) DESC",

    # ── ORDER BY no LIMIT ──
    "SELECT p.title, p.score, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id ORDER BY p.view_count DESC",
    "SELECT p.title, p.score FROM posts p JOIN users u ON p.owner_user_id = u.id ORDER BY u.reputation DESC",
    "SELECT id, title, score, view_count FROM posts ORDER BY view_count DESC",
    "SELECT id, display_name, reputation FROM users ORDER BY reputation DESC",
    "SELECT id, display_name, reputation, location FROM users ORDER BY reputation DESC",
    "SELECT id, title, answer_count FROM posts ORDER BY answer_count DESC",
    "SELECT id, title, score FROM posts ORDER BY score ASC",
    "SELECT id, display_name FROM users ORDER BY creation_date DESC",
    "SELECT id, title FROM posts ORDER BY creation_date ASC",
    "SELECT id, post_id, vote_type_id FROM votes ORDER BY creation_date DESC",

    # ── OR conditions ──
    "SELECT * FROM users WHERE location LIKE '%United States%' OR location LIKE '%Canada%' OR location LIKE '%United Kingdom%'",
    "SELECT * FROM users WHERE location LIKE '%Germany%' OR location LIKE '%France%' OR location LIKE '%Italy%'",
    "SELECT * FROM posts WHERE score > 50 OR view_count > 10000 OR answer_count > 10",
    "SELECT * FROM posts WHERE score > 100 OR view_count > 50000",
    "SELECT * FROM votes WHERE vote_type_id = 1 OR vote_type_id = 2 OR vote_type_id = 3 OR vote_type_id = 4",
    "SELECT * FROM users WHERE location LIKE '%Australia%' OR location LIKE '%New Zealand%'",
    "SELECT * FROM posts WHERE answer_count = 0 OR score < 0",
    "SELECT * FROM users WHERE reputation > 10000 OR reputation < 0",
    "SELECT * FROM posts WHERE post_type_id = 1 OR post_type_id = 2 OR post_type_id = 3",
    "SELECT * FROM users WHERE location LIKE '%Spain%' OR location LIKE '%Portugal%' OR location LIKE '%Mexico%'",
    "SELECT * FROM posts WHERE score > 100 OR answer_count > 20 OR view_count > 100000",
    "SELECT * FROM users WHERE location LIKE '%New York%' OR location LIKE '%Los Angeles%' OR location LIKE '%Chicago%'",
    "SELECT * FROM posts WHERE post_type_id = 1 OR post_type_id = 2 OR score > 50",
    "SELECT * FROM users WHERE reputation > 5000 OR reputation < 10",
    "SELECT * FROM votes WHERE vote_type_id = 1 OR vote_type_id = 5 OR vote_type_id = 8",
    "SELECT p.title FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.location LIKE '%United States%' OR u.location LIKE '%Canada%'",
    "SELECT * FROM posts WHERE score < 0 OR answer_count = 0 OR view_count < 10",
    "SELECT * FROM users WHERE display_name LIKE '%test%' OR display_name LIKE '%admin%' OR display_name LIKE '%user%'",
    "SELECT * FROM posts WHERE title LIKE '%help%' OR title LIKE '%error%' OR title LIKE '%problem%'",
    "SELECT * FROM posts WHERE creation_date < '2012-01-01' OR creation_date > '2023-01-01'",

    # ── functions in WHERE ──
    "SELECT * FROM users WHERE LOWER(display_name) = 'john'",
    "SELECT * FROM users WHERE LOWER(display_name) = 'smith'",
    "SELECT * FROM users WHERE UPPER(location) = 'UNITED STATES'",
    "SELECT * FROM posts WHERE EXTRACT(YEAR FROM creation_date) = 2020",
    "SELECT * FROM posts WHERE EXTRACT(YEAR FROM creation_date) = 2019",
    "SELECT * FROM posts WHERE EXTRACT(MONTH FROM creation_date) = 6",
    "SELECT * FROM users WHERE LENGTH(location) > 20",
    "SELECT * FROM users WHERE LENGTH(display_name) > 15",
    "SELECT * FROM posts WHERE LENGTH(title) > 100",
    "SELECT * FROM posts WHERE EXTRACT(YEAR FROM creation_date) = 2018",
    "SELECT * FROM posts WHERE EXTRACT(YEAR FROM creation_date) = 2021",
    "SELECT * FROM posts WHERE EXTRACT(YEAR FROM creation_date) = 2022",
    "SELECT * FROM posts WHERE EXTRACT(MONTH FROM creation_date) = 1",
    "SELECT * FROM posts WHERE EXTRACT(MONTH FROM creation_date) = 12",
    "SELECT * FROM users WHERE LOWER(display_name) = 'anonymous'",
    "SELECT * FROM users WHERE LOWER(display_name) LIKE '%user%'",
    "SELECT * FROM users WHERE LENGTH(location) > 30",
    "SELECT * FROM posts WHERE LENGTH(title) > 80",
    "SELECT * FROM users WHERE UPPER(location) LIKE '%UNITED%'",
    "SELECT * FROM posts WHERE EXTRACT(DAY FROM creation_date) = 1",

    # ── DISTINCT ──
    "SELECT DISTINCT location FROM users",
    "SELECT DISTINCT post_type_id FROM posts",
    "SELECT DISTINCT vote_type_id FROM votes",
    "SELECT DISTINCT owner_user_id FROM posts",
    "SELECT DISTINCT u.location, p.post_type_id FROM users u JOIN posts p ON p.owner_user_id = u.id",
    "SELECT DISTINCT location FROM users ORDER BY location",
    "SELECT DISTINCT display_name FROM users ORDER BY display_name",
    "SELECT DISTINCT score FROM posts ORDER BY score DESC",
    "SELECT DISTINCT answer_count FROM posts ORDER BY answer_count DESC",
    "SELECT DISTINCT u.location, p.post_type_id, AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.location, p.post_type_id",
    "SELECT DISTINCT owner_user_id FROM posts WHERE score > 10",
    "SELECT DISTINCT post_id FROM votes WHERE vote_type_id = 2",
    "SELECT DISTINCT u.display_name FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE p.score > 50",
    "SELECT DISTINCT u.location FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE p.score > 100",
    "SELECT DISTINCT p.post_type_id FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.reputation > 1000",

    # ── heavy multi table no LIMIT ──
    "SELECT p.title, p.score, p.view_count, u.display_name, u.location, u.reputation FROM posts p JOIN users u ON p.owner_user_id = u.id ORDER BY p.score DESC",
    "SELECT p.title, p.score, u.display_name, COUNT(v.id) as votes FROM posts p JOIN users u ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY p.title, p.score, u.display_name ORDER BY votes DESC",
    "SELECT u.display_name, u.reputation, u.location, COUNT(p.id), AVG(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.display_name, u.reputation, u.location ORDER BY u.reputation DESC",
    "SELECT p.title, p.score, p.view_count, p.answer_count, u.display_name, u.reputation FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.score > 10 ORDER BY p.view_count DESC",
    "SELECT u.display_name, p.title, p.score, v.vote_type_id FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id WHERE u.reputation > 500 ORDER BY p.score DESC",
    "SELECT u.location, COUNT(p.id), AVG(p.score), AVG(u.reputation) FROM users u JOIN posts p ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY u.location ORDER BY COUNT(p.id) DESC",
    "SELECT p.title, COUNT(v.id), AVG(p.score) FROM posts p JOIN votes v ON v.post_id = p.id GROUP BY p.title ORDER BY COUNT(v.id) DESC",
    "SELECT u.display_name, u.location, SUM(p.view_count), COUNT(p.id) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.location LIKE '%United States%' GROUP BY u.display_name, u.location ORDER BY SUM(p.view_count) DESC",
    "SELECT p.post_type_id, COUNT(p.id), AVG(p.score), SUM(p.view_count) FROM posts p JOIN users u ON p.owner_user_id = u.id JOIN votes v ON v.post_id = p.id GROUP BY p.post_type_id ORDER BY SUM(p.view_count) DESC",
    "SELECT u.display_name, COUNT(p.id), SUM(p.score), AVG(p.view_count), MAX(p.score) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.reputation > 100 GROUP BY u.display_name ORDER BY SUM(p.score) DESC",

    # ══════════════════════════════════════
    # FAST QUERIES
    # ══════════════════════════════════════

    # ── lookup by primary key ──
    "SELECT id, title, score FROM posts WHERE id = 100",
    "SELECT id, title, score FROM posts WHERE id = 500",
    "SELECT id, title, score FROM posts WHERE id = 1000",
    "SELECT id, title, score FROM posts WHERE id = 5000",
    "SELECT id, title, score FROM posts WHERE id = 10000",
    "SELECT id, title, score FROM posts WHERE id = 20000",
    "SELECT id, title, score FROM posts WHERE id = 50000",
    "SELECT id, title, score FROM posts WHERE id = 100000",
    "SELECT id, display_name, reputation FROM users WHERE id = 10",
    "SELECT id, display_name, reputation FROM users WHERE id = 100",
    "SELECT id, display_name, reputation FROM users WHERE id = 500",
    "SELECT id, display_name, reputation FROM users WHERE id = 1000",
    "SELECT id, display_name, reputation FROM users WHERE id = 5000",
    "SELECT id, display_name, reputation FROM users WHERE id = 10000",
    "SELECT id, display_name, reputation FROM users WHERE id = 50000",
    "SELECT id, post_id, vote_type_id FROM votes WHERE id = 100",
    "SELECT id, post_id, vote_type_id FROM votes WHERE id = 1000",
    "SELECT id, post_id, vote_type_id FROM votes WHERE id = 10000",
    "SELECT id, post_id, vote_type_id FROM votes WHERE id = 100000",
    "SELECT id, post_id, vote_type_id FROM votes WHERE id = 500000",

    # ── simple COUNT ──
    "SELECT COUNT(*) FROM posts WHERE post_type_id = 1",
    "SELECT COUNT(*) FROM posts WHERE post_type_id = 2",
    "SELECT COUNT(*) FROM votes WHERE post_id = 500",
    "SELECT COUNT(*) FROM votes WHERE post_id = 1000",
    "SELECT COUNT(*) FROM votes WHERE vote_type_id = 2",
    "SELECT COUNT(*) FROM users WHERE reputation > 10000",
    "SELECT COUNT(*) FROM posts WHERE score > 100",
    "SELECT COUNT(*) FROM posts WHERE answer_count = 0",
    "SELECT COUNT(*) FROM posts WHERE score < 0",
    "SELECT COUNT(*) FROM votes WHERE vote_type_id = 1",
    "SELECT COUNT(*) FROM votes WHERE vote_type_id = 3",
    "SELECT COUNT(*) FROM votes WHERE vote_type_id = 5",
    "SELECT COUNT(*) FROM posts WHERE score = 0",
    "SELECT COUNT(*) FROM users WHERE reputation = 1",
    "SELECT COUNT(*) FROM posts WHERE answer_count > 10",
    "SELECT COUNT(*) FROM posts WHERE view_count > 10000",
    "SELECT COUNT(*) FROM users WHERE reputation > 5000",
    "SELECT COUNT(*) FROM posts WHERE score > 50",
    "SELECT COUNT(*) FROM posts WHERE post_type_id = 1 AND score > 10",
    "SELECT COUNT(*) FROM votes WHERE vote_type_id = 2 AND post_id < 10000",

    # ── specific columns with LIMIT ──
    "SELECT id, title, score FROM posts LIMIT 10",
    "SELECT id, title, score FROM posts LIMIT 20",
    "SELECT id, title, score FROM posts LIMIT 50",
    "SELECT id, display_name, reputation FROM users LIMIT 10",
    "SELECT id, display_name, reputation FROM users LIMIT 20",
    "SELECT id, display_name, location FROM users LIMIT 50",
    "SELECT id, post_id, vote_type_id FROM votes LIMIT 10",
    "SELECT id, post_id, vote_type_id FROM votes LIMIT 20",
    "SELECT id, title, score, view_count FROM posts LIMIT 100",
    "SELECT id, display_name, reputation FROM users LIMIT 100",
    "SELECT id, title FROM posts LIMIT 5",
    "SELECT id, display_name FROM users LIMIT 5",
    "SELECT id, post_id FROM votes LIMIT 5",
    "SELECT id, title, answer_count FROM posts LIMIT 10",
    "SELECT id, title, creation_date FROM posts LIMIT 10",
    "SELECT id, display_name, creation_date FROM users LIMIT 10",
    "SELECT id, score, view_count FROM posts LIMIT 20",
    "SELECT id, reputation, location FROM users LIMIT 20",
    "SELECT id, vote_type_id, creation_date FROM votes LIMIT 20",
    "SELECT id, title, post_type_id FROM posts LIMIT 50",

    # ── filtered with LIMIT ──
    "SELECT id, title, score FROM posts WHERE score > 50 LIMIT 10",
    "SELECT id, title, score FROM posts WHERE score > 100 LIMIT 20",
    "SELECT id, title, score FROM posts WHERE score > 200 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation > 1000 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation > 5000 LIMIT 20",
    "SELECT id, title FROM posts WHERE answer_count > 5 LIMIT 10",
    "SELECT id, title FROM posts WHERE view_count > 1000 LIMIT 10",
    "SELECT id, title FROM posts WHERE post_type_id = 1 LIMIT 20",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 2 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 1 LIMIT 20",
    "SELECT id, title FROM posts WHERE score > 500 LIMIT 5",
    "SELECT id, display_name FROM users WHERE reputation > 20000 LIMIT 5",
    "SELECT id, title FROM posts WHERE answer_count > 20 LIMIT 5",
    "SELECT id, title FROM posts WHERE score > 50 AND answer_count > 2 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation > 1000 AND reputation < 5000 LIMIT 10",
    "SELECT id, title FROM posts WHERE post_type_id = 2 AND score > 10 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 3 LIMIT 10",
    "SELECT id, title FROM posts WHERE view_count > 5000 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation BETWEEN 1000 AND 2000 LIMIT 10",
    "SELECT id, title FROM posts WHERE creation_date > '2021-01-01' LIMIT 10",

    # ── JOIN with LIMIT ──
    "SELECT p.title, p.score, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.score > 50 LIMIT 10",
    "SELECT p.title, p.score, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.score > 100 LIMIT 20",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.reputation > 1000 LIMIT 10",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id LIMIT 10",
    "SELECT p.id, p.score, v.vote_type_id FROM posts p JOIN votes v ON v.post_id = p.id WHERE p.score > 100 LIMIT 10",
    "SELECT p.title, COUNT(v.id) as vote_count FROM posts p JOIN votes v ON v.post_id = p.id WHERE p.score > 50 GROUP BY p.title LIMIT 10",
    "SELECT u.display_name, COUNT(p.id) FROM users u JOIN posts p ON p.owner_user_id = u.id GROUP BY u.display_name LIMIT 10",
    "SELECT p.title, p.score FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.score > 200 ORDER BY p.score DESC LIMIT 10",
    "SELECT p.title, p.score FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.answer_count > 3 LIMIT 20",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.post_type_id = 1 LIMIT 10",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.score > 500 LIMIT 5",
    "SELECT p.title, u.reputation FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.reputation > 5000 LIMIT 10",
    "SELECT p.id, p.score FROM posts p JOIN votes v ON v.post_id = p.id WHERE v.vote_type_id = 2 LIMIT 10",
    "SELECT p.title, p.answer_count FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.answer_count > 5 LIMIT 10",
    "SELECT u.display_name, p.score FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.reputation > 2000 LIMIT 10",
    "SELECT p.title, p.score FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.score > 100 AND u.reputation > 500 LIMIT 10",
    "SELECT p.title, p.view_count FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.view_count > 500 LIMIT 10",
    "SELECT u.display_name, COUNT(p.id) FROM users u JOIN posts p ON p.owner_user_id = u.id WHERE u.reputation > 1000 GROUP BY u.display_name LIMIT 10",
    "SELECT p.title, u.display_name FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE p.creation_date > '2021-01-01' LIMIT 10",
    "SELECT p.id, v.vote_type_id FROM posts p JOIN votes v ON v.post_id = p.id WHERE p.score > 50 LIMIT 20",

    # ── ORDER BY with LIMIT ──
    "SELECT id, title, score FROM posts ORDER BY score DESC LIMIT 10",
    "SELECT id, title, score FROM posts ORDER BY score DESC LIMIT 20",
    "SELECT id, display_name, reputation FROM users ORDER BY reputation DESC LIMIT 10",
    "SELECT id, display_name, reputation FROM users ORDER BY reputation DESC LIMIT 20",
    "SELECT id, title, view_count FROM posts ORDER BY view_count DESC LIMIT 10",
    "SELECT id, title, answer_count FROM posts ORDER BY answer_count DESC LIMIT 10",
    "SELECT id, display_name FROM users ORDER BY reputation DESC LIMIT 5",
    "SELECT id, title FROM posts ORDER BY creation_date DESC LIMIT 10",
    "SELECT id, display_name FROM users ORDER BY creation_date DESC LIMIT 10",
    "SELECT id, post_id FROM votes ORDER BY creation_date DESC LIMIT 10",
    "SELECT id, title, score FROM posts ORDER BY score ASC LIMIT 10",
    "SELECT id, title FROM posts ORDER BY view_count DESC LIMIT 5",
    "SELECT id, display_name FROM users ORDER BY reputation ASC LIMIT 10",
    "SELECT id, title FROM posts ORDER BY answer_count DESC LIMIT 5",
    "SELECT id, title FROM posts ORDER BY score DESC LIMIT 100",
    "SELECT id, display_name FROM users ORDER BY reputation DESC LIMIT 100",
    "SELECT id, title, score FROM posts WHERE score > 10 ORDER BY score DESC LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation > 100 ORDER BY reputation DESC LIMIT 10",
    "SELECT id, title FROM posts WHERE answer_count > 0 ORDER BY answer_count DESC LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 2 ORDER BY creation_date DESC LIMIT 10",

    # ── simple aggregation with filter ──
    "SELECT AVG(score) FROM posts WHERE post_type_id = 1",
    "SELECT MAX(score) FROM posts WHERE post_type_id = 1",
    "SELECT MIN(score) FROM posts WHERE score > 0",
    "SELECT AVG(reputation) FROM users WHERE reputation > 0",
    "SELECT MAX(reputation) FROM users",
    "SELECT MIN(creation_date) FROM posts",
    "SELECT MAX(creation_date) FROM posts",
    "SELECT COUNT(*), AVG(score) FROM posts WHERE answer_count > 0",
    "SELECT COUNT(*) FROM posts WHERE creation_date > '2021-01-01'",
    "SELECT COUNT(*) FROM users WHERE creation_date > '2020-01-01'",
    "SELECT AVG(score) FROM posts WHERE score > 0",
    "SELECT MAX(view_count) FROM posts WHERE post_type_id = 1",
    "SELECT MIN(reputation) FROM users WHERE reputation > 0",
    "SELECT AVG(answer_count) FROM posts WHERE post_type_id = 1",
    "SELECT COUNT(*), AVG(reputation) FROM users WHERE reputation > 100",
    "SELECT MAX(score), MIN(score) FROM posts WHERE post_type_id = 2",
    "SELECT COUNT(*) FROM posts WHERE score > 0 AND answer_count > 0",
    "SELECT AVG(view_count) FROM posts WHERE score > 10",
    "SELECT COUNT(*) FROM votes WHERE creation_date > '2021-01-01'",
    "SELECT MAX(creation_date) FROM users",

    # ── exact match lookups ──
    "SELECT id, title FROM posts WHERE score = 0 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation = 1 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 3 LIMIT 10",
    "SELECT id, title FROM posts WHERE answer_count = 1 LIMIT 20",
    "SELECT id, title FROM posts WHERE post_type_id = 2 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation = 101 LIMIT 10",
    "SELECT id, title FROM posts WHERE view_count = 0 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 5 LIMIT 10",
    "SELECT id, title FROM posts WHERE score = 1 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation = 11 LIMIT 10",
    "SELECT id, title FROM posts WHERE answer_count = 0 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 4 LIMIT 10",
    "SELECT id, title FROM posts WHERE score = 2 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation = 21 LIMIT 10",
    "SELECT id, title FROM posts WHERE post_type_id = 1 AND score = 5 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation = 51 LIMIT 10",
    "SELECT id, title FROM posts WHERE answer_count = 2 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 1 AND post_id < 1000 LIMIT 10",
    "SELECT id, title FROM posts WHERE score = 10 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation = 201 LIMIT 10",

    # ── simple single table no JOIN ──
    "SELECT id, title, score FROM posts WHERE score > 50 AND answer_count > 0 LIMIT 20",
    "SELECT id, display_name, reputation FROM users WHERE reputation > 500 AND reputation < 1000 LIMIT 20",
    "SELECT id, title FROM posts WHERE post_type_id = 1 AND answer_count > 3 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation > 100 LIMIT 50",
    "SELECT id, title, view_count FROM posts WHERE view_count > 100 AND score > 5 LIMIT 10",
    "SELECT id, title FROM posts WHERE score > 20 AND score < 50 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation > 200 AND reputation < 500 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id = 2 AND post_id > 100 LIMIT 10",
    "SELECT id, title FROM posts WHERE answer_count BETWEEN 1 AND 5 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation BETWEEN 100 AND 500 LIMIT 10",
    "SELECT id, title, score FROM posts WHERE score > 0 ORDER BY score DESC LIMIT 5",
    "SELECT id, display_name FROM users WHERE reputation > 1000 ORDER BY reputation DESC LIMIT 5",
    "SELECT id, title FROM posts WHERE creation_date > '2022-01-01' LIMIT 10",
    "SELECT id, display_name FROM users WHERE creation_date > '2021-01-01' LIMIT 10",
    "SELECT id, post_id FROM votes WHERE creation_date > '2022-01-01' LIMIT 10",
    "SELECT id, title FROM posts WHERE score > 100 AND view_count > 100 LIMIT 10",
    "SELECT id, display_name FROM users WHERE reputation > 500 LIMIT 30",
    "SELECT id, title FROM posts WHERE post_type_id = 2 AND answer_count = 0 LIMIT 10",
    "SELECT id, post_id FROM votes WHERE vote_type_id IN (1, 2) LIMIT 10",
    "SELECT id, title FROM posts WHERE score BETWEEN 50 AND 200 LIMIT 10",
]

def extract_features(query):
    q = query.upper()
    return {
        "has_select_star": 1 if "SELECT *" in q else 0,
        "has_like_wildcard": 1 if "LIKE '%" in q else 0,
        "join_count": q.count("JOIN"),
        "has_subquery": 1 if q.count("SELECT") > 1 else 0,
        "has_group_by": 1 if "GROUP BY" in q else 0,
        "has_order_by_no_limit": 1 if "ORDER BY" in q and "LIMIT" not in q else 0,
        "has_or": 1 if " OR " in q else 0,
        "has_function_in_where": 1 if any(f in q for f in ["LOWER(", "UPPER(", "EXTRACT(", "LENGTH("]) else 0,
    }

results = []
n = len(queries)

#queries = queries[:5] # limit to 5 for quick testing; replace with `queries` for full run

for i, query in enumerate(queries, start=1):
    print(f"[{i}/{n}] running... ", end="", flush=True)
    m = measure_query(conn, query)
    if m is None:
        print("ERROR — skipped")
        continue

    cold, warm_median, warm_std, all_times, timed_out_count = m
    features = extract_features(query)
    label = "slow" if warm_median > SLOW_THRESHOLD_MS else "fast"

    results.append({
        "query_id": i,
        "query_text": query,
        "execution_time_cold_ms": round(cold, 2),
        "execution_time_warm_ms": round(warm_median, 2),
        "execution_time_warm_std_ms": round(warm_std, 2),
        "all_runs_ms": ";".join(f"{t:.2f}" for t in all_times),
        "timed_out_runs": timed_out_count,
        "label": label,
        **features,
    })
    suffix = f"  [{timed_out_count} timeout(s)]" if timed_out_count > 0 else ""
    print(f"cold={cold:.1f}ms  warm_median={warm_median:.1f}ms  → {label}{suffix}")

fieldnames = [
    "query_id", "query_text",
    "execution_time_cold_ms", "execution_time_warm_ms",
    "execution_time_warm_std_ms", "all_runs_ms",
    "timed_out_runs",
    "label",
    "has_select_star", "has_like_wildcard", "join_count",
    "has_subquery", "has_group_by", "has_order_by_no_limit",
    "has_or", "has_function_in_where",
]

with open("training_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    writer.writerows(results)

slow = sum(1 for r in results if r["label"] == "slow")
fast = sum(1 for r in results if r["label"] == "fast")
print(f"\nDone! Slow: {slow}  Fast: {fast}  Total: {len(results)}")

cur.close()
conn.close()