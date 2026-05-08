"""One-shot pipeline test — runs a known slow query through the full flow."""
import psycopg2
from monitor.monitor import (
    process_one_query, get_table_schema, DB_CONFIG
)
from classifier.inference import QueryAnalyzer
from finetune.llm_optimizer import LLMOptimizer

conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = True
schema = get_table_schema(conn)
analyzer = QueryAnalyzer()
optimizer = LLMOptimizer()

test_query = ("SELECT u.location, AVG(p.score) "
              "FROM users u JOIN posts p ON p.owner_user_id = u.id "
              "JOIN votes v ON v.post_id = p.id "
              "GROUP BY u.location ORDER BY AVG(p.score) DESC")

process_one_query(test_query, mean_time_ms=350.0, calls=1,
                  analyzer=analyzer, conn=conn, schema=schema,
                  optimizer=optimizer)

conn.close()