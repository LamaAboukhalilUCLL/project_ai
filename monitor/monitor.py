import time
import pickle
import re
import sys
import os
import json
import datetime
import psycopg2
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embeddings.embeddings import find_similar, store_fix

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB_CONFIG = {
    "dbname": "stackexchange_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432"
}

SLOW_THRESHOLD_MS = 100
POLL_INTERVAL_SECONDS = 5
LOG_FILE = "dashboard/query_log.json"

class QueryClassifier(nn.Module):
    def __init__(self):
        super(QueryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

def load_classifier():
    model = QueryClassifier()
    model.load_state_dict(torch.load("classifier/query_classifier.pth", weights_only=True))
    model.eval()
    with open("classifier/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def extract_features(query):
    q = query.upper()
    return np.array([[
        1 if "SELECT *" in q else 0,
        1 if "LIKE '%" in q else 0,
        q.count("JOIN"),
        1 if q.count("SELECT") > 1 else 0,
        1 if "GROUP BY" in q else 0,
        1 if "ORDER BY" in q and "LIMIT" not in q else 0,
        1 if " OR " in q else 0,
        1 if any(f in q for f in ["LOWER(", "UPPER(", "EXTRACT(", "LENGTH("]) else 0,
    ]], dtype=np.float32)

def is_slow_by_classifier(query, model, scaler):
    features = extract_features(query)
    features_scaled = scaler.transform(features)
    tensor = torch.tensor(features_scaled, dtype=torch.float32)
    with torch.no_grad():
        prob = model(tensor).item()
    return prob > 0.5, prob

def get_table_schema(conn):
    cur = conn.cursor()
    schema = ""
    for table in ["posts", "users", "votes"]:
        cur.execute(f"""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = '{table}'
            ORDER BY ordinal_position
        """)
        cols = cur.fetchall()
        schema += f"\nTable: {table}\n"
        for col_name, col_type in cols:
            schema += f"  - {col_name}: {col_type}\n"
    cur.close()
    return schema

def ask_gpt(slow_query, schema, similar_fix=None):
    past_example = ""
    if similar_fix:
        past_example = f"""
A similar slow query was fixed before:
Slow query: {similar_fix['slow_query']}
Fix applied: {similar_fix['fix']}
Speedup achieved: {similar_fix['speedup_ms']}ms faster
"""

    prompt = f"""You are a PostgreSQL performance expert.

A slow SQL query was detected. Analyze it and suggest an optimization.

Database schema:
{schema}

Slow query:
{slow_query}

{past_example}

Respond in this exact format:
REASON: [one sentence explaining why this query is slow]
FIX: [the optimized SQL query only, no explanation]
INDEX: [optional CREATE INDEX statement if needed, or NONE]
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

def run_and_time(conn, query):
    cur = conn.cursor()
    try:
        cur.execute("SET statement_timeout = '10s'")
        cur.execute(f"EXPLAIN ANALYZE {query}")
        rows = cur.fetchall()
        for row in rows:
            if "Execution Time:" in row[0]:
                match = re.search(r"Execution Time: ([\d.]+)", row[0])
                if match:
                    return float(match.group(1))
    except Exception:
        conn.rollback()
        return None
    finally:
        cur.close()
    return None

def get_recent_slow_queries(conn):
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT query, mean_exec_time, calls
            FROM pg_stat_statements
            WHERE mean_exec_time > %s
            AND query NOT LIKE '%%pg_stat_statements%%'
            AND query NOT LIKE '%%EXPLAIN%%'
            AND query NOT LIKE '%%SET%%'
            AND query ILIKE ANY(ARRAY['%%SELECT%%'])
            ORDER BY mean_exec_time DESC
            LIMIT 10
        """, (SLOW_THRESHOLD_MS,))
        return cur.fetchall()
    except Exception as e:
        conn.rollback()
        print(f"Error reading pg_stat_statements: {e}")
        return []
    finally:
        cur.close()

def log_query(query, avg_time_ms, confidence, reason, fix, speedup_ms):
    log = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            log = json.load(f)
    log.append({
        "query": query[:200],
        "avg_time_ms": round(avg_time_ms, 1),
        "confidence": round(confidence * 100, 1),
        "reason": reason,
        "fix": fix,
        "speedup_ms": round(speedup_ms, 1) if speedup_ms else 0,
        "flagged": True,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)

def main():
    print("=" * 60)
    print("  SQL Query Monitor — starting up")
    print("=" * 60)

    print("Loading classifier...")
    model, scaler = load_classifier()
    print("Classifier loaded.")

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("Connected.")

    schema = get_table_schema(conn)
    seen_queries = set()

    print(f"\nMonitoring for slow queries (threshold: {SLOW_THRESHOLD_MS}ms)")
    print(f"Polling every {POLL_INTERVAL_SECONDS} seconds")
    print("-" * 60)

    while True:
        slow_queries = get_recent_slow_queries(conn)

        for query, mean_time, calls in slow_queries:
            query_key = query[:100]
            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)

            is_slow, confidence = is_slow_by_classifier(query, model, scaler)
            if not is_slow:
                continue

            print(f"\n[SLOW QUERY DETECTED]")
            print(f"Query:      {query[:120]}...")
            print(f"Avg time:   {mean_time:.1f}ms")
            print(f"Calls:      {calls}")
            print(f"Confidence: {confidence*100:.1f}% slow")

            similar = find_similar(query)

            print("Asking GPT for optimization...")
            gpt_response = ask_gpt(query, schema, similar)
            print(f"\n{gpt_response}")

            reason = ""
            fix_query = ""
            speedup = 0

            reason_match = re.search(r"REASON:\s*(.+?)(?=FIX:|$)", gpt_response, re.DOTALL)
            if reason_match:
                reason = reason_match.group(1).strip()

            fix_match = re.search(r"FIX:\s*(.+?)(?=INDEX:|$)", gpt_response, re.DOTALL)
            if fix_match:
                fix_query = fix_match.group(1).strip()
                optimized_time = run_and_time(conn, fix_query)
                if optimized_time:
                    speedup = mean_time - optimized_time
                    print(f"\nOriginal:  {mean_time:.1f}ms")
                    print(f"Optimized: {optimized_time:.1f}ms")
                    print(f"Speedup:   {speedup:.1f}ms faster")
                    if speedup > 0:
                        store_fix(query, fix_query, speedup)
                        print("Fix stored in history.")

            log_query(
                query=query,
                avg_time_ms=mean_time,
                confidence=confidence,
                reason=reason,
                fix=fix_query,
                speedup_ms=speedup
            )

            print("-" * 60)

        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    main()