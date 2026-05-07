"""
Inference wrapper around the trained multi-task query model.

Provides a single QueryAnalyzer class with two methods:
  - analyze(query, conn=None): returns {p_slow, predicted_ms, features}
  - rank_candidates(original, candidates, conn=None): returns
    a list of {query, p_slow, predicted_ms, predicted_speedup_ms}
    sorted by predicted speedup (best first).

Plan features require a DB connection. When conn is None we fall back
to the text-only features and pad plan features with zeros — useful
for unit-testing or when the DB is briefly unavailable.
"""

import json
import os
import pickle
import re

import numpy as np
import psycopg2
import psycopg2.errors
import torch
import torch.nn as nn


# ─────────────────────────── Model definition ───────────────────────────
# Must match the architecture in classifier/classifier.py exactly,
# otherwise load_state_dict will fail.

class MultiTaskQueryModel(nn.Module):
    def __init__(self, n_features=20, dropout=0.3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_clf = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        z = self.trunk(x)
        return self.head_clf(z), self.head_reg(z)


# ─────────────────────────── Feature extraction ───────────────────────────

DEFAULT_PLAN_FEATURES = {
    "plan_total_cost": 0.0,
    "plan_rows": 0,
    "actual_rows": 0,
    "plan_depth": 0,
    "shared_hit": 0,
    "shared_read": 0,
    "has_seq_scan": 0,
    "has_index_scan": 0,
    "has_bitmap_scan": 0,
    "has_hash_join": 0,
    "has_nested_loop": 0,
    "has_merge_join": 0,
}


def extract_text_features(query):
    """Cheap regex-derived features. Always available."""
    q = query.upper()
    return {
        "has_select_star": 1 if "SELECT *" in q else 0,
        "has_like_wildcard": 1 if "LIKE '%" in q else 0,
        "join_count": q.count("JOIN"),
        "has_subquery": 1 if q.count("SELECT") > 1 else 0,
        "has_group_by": 1 if "GROUP BY" in q else 0,
        "has_order_by_no_limit": 1 if "ORDER BY" in q and "LIMIT" not in q else 0,
        "has_or": 1 if " OR " in q else 0,
        "has_function_in_where": 1 if any(
            f in q for f in ["LOWER(", "UPPER(", "EXTRACT(", "LENGTH("]
        ) else 0,
    }


def replace_placeholders(query):
    """
    pg_stat_statements returns queries like 'WHERE id = $1'. EXPLAIN
    can't bind those, so we substitute literal placeholders. Heuristic:
      - Numeric-looking context  -> 1
      - String-looking context   -> 'x'
      - Default                  -> 1
    Good enough to get a plan; the *plan shape* matters more than the
    specific value for our features.
    """
    # Replace $1, $2, ... with literals
    def _sub(match):
        # If the placeholder is inside quotes already, leave it
        return "1"
    return re.sub(r"\$\d+", _sub, query)


def extract_plan_features(conn, query, timeout_ms=5000):
    """
    Run EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) against `query` and walk
    the plan tree. Returns the same 12-feature dict as in training.
    On any error (including timeouts) returns the default zeros.

    Note: this actually runs the query. For monitor use that's fine —
    we already know it's a slow candidate we want analysed.
    """
    feats = dict(DEFAULT_PLAN_FEATURES)
    safe_query = replace_placeholders(query)

    cur = conn.cursor()
    try:
        cur.execute(f"SET statement_timeout = '{timeout_ms}ms'")
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {safe_query}")
        result = cur.fetchone()[0]
        if isinstance(result, str):
            result = json.loads(result)
        plan = result[0]["Plan"]
        feats["plan_total_cost"] = plan.get("Total Cost", 0.0)
        feats["plan_rows"] = plan.get("Plan Rows", 0)
        feats["actual_rows"] = plan.get("Actual Rows", 0)

        def walk(node, depth=0):
            feats["plan_depth"] = max(feats["plan_depth"], depth)
            feats["shared_hit"] += node.get("Shared Hit Blocks", 0)
            feats["shared_read"] += node.get("Shared Read Blocks", 0)
            nt = node.get("Node Type", "")
            if "Seq Scan" in nt: feats["has_seq_scan"] = 1
            if "Index Scan" in nt or "Index Only Scan" in nt: feats["has_index_scan"] = 1
            if "Bitmap" in nt: feats["has_bitmap_scan"] = 1
            if "Hash Join" in nt: feats["has_hash_join"] = 1
            if "Nested Loop" in nt: feats["has_nested_loop"] = 1
            if "Merge Join" in nt: feats["has_merge_join"] = 1
            for child in node.get("Plans", []):
                walk(child, depth + 1)

        walk(plan)
    except Exception:
        # Could be QueryCanceled (timeout), syntax error from the
        # placeholder substitution, etc. Default zeros are safer than
        # crashing the monitor loop.
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        cur.close()
    return feats


# ─────────────────────────── Analyzer ───────────────────────────

class QueryAnalyzer:
    """
    Loads the multi-task model + scaler + metadata, exposes an analyze()
    method, and a rank_candidates() helper for the model-in-the-loop
    optimization pipeline.
    """

    def __init__(self,
                 model_path="classifier/query_classifier.pth",
                 scaler_path="classifier/scaler.pkl",
                 meta_path="classifier/model_meta.json"):
        with open(meta_path, "r") as f:
            self.meta = json.load(f)
        self.feature_cols = self.meta["feature_cols"]

        self.model = MultiTaskQueryModel(n_features=self.meta["n_features"])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

    def _build_feature_vector(self, conn, query):
        text = extract_text_features(query)
        plan = (extract_plan_features(conn, query)
                if conn is not None else dict(DEFAULT_PLAN_FEATURES))
        merged = {**text, **plan}
        # Order MUST match training. We rely on meta["feature_cols"].
        vec = np.array([[merged[c] for c in self.feature_cols]],
                       dtype=np.float32)
        return vec, merged

    def analyze(self, query, conn=None):
        """Returns {p_slow, predicted_ms, features}."""
        vec, features = self._build_feature_vector(conn, query)
        scaled = self.scaler.transform(vec)
        x = torch.tensor(scaled, dtype=torch.float32)
        with torch.no_grad():
            logit, log_ms = self.model(x)
            p_slow = torch.sigmoid(logit).item()
            # log1p(10000) ≈ 9.21; we clamp at 12 (~163s) to be safe against
            # extrapolation while leaving room for genuine very-slow queries.
            log_ms_clamped = float(np.clip(log_ms.item(), -1.0, 12.0))
            predicted_ms = float(np.expm1(log_ms_clamped))
        return {
            "p_slow": p_slow,
            "predicted_ms": predicted_ms,
            "features": features,
        }

    def rank_candidates(self, original_query, candidates, conn=None):
        """
        Score the original + each candidate with the regression head,
        return a list sorted by predicted speedup vs original (best first).
        Each entry: {query, p_slow, predicted_ms, predicted_speedup_ms}.
        """
        baseline = self.analyze(original_query, conn=conn)
        baseline_ms = baseline["predicted_ms"]

        scored = []
        for c in candidates:
            r = self.analyze(c, conn=conn)
            scored.append({
                "query": c,
                "p_slow": r["p_slow"],
                "predicted_ms": r["predicted_ms"],
                "predicted_speedup_ms": baseline_ms - r["predicted_ms"],
            })
        scored.sort(key=lambda d: d["predicted_speedup_ms"], reverse=True)
        return baseline, scored


# ─────────────────────────── CLI smoke test ───────────────────────────

if __name__ == "__main__":
    """
    Quick sanity check. Run from project root:
        python -m classifier.inference
    """
    DB_CONFIG = {
        "dbname": "stackexchange_db",
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432",
    }
    sample = "SELECT * FROM users WHERE LOWER(display_name) = 'smith'"

    print("Loading analyzer...")
    az = QueryAnalyzer()
    print(f"  model expects {az.meta['n_features']} features")
    final = az.meta.get("final_metrics")
    if final and "test_accuracy" in final:
        print(f"  test accuracy from training: "
              f"{final['test_accuracy']*100:.1f}%")
    else:
        print("  (no final_metrics in model_meta.json — re-run the notebook "
              "or classifier.py to populate)")

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True

    print(f"\nAnalyzing: {sample}")
    result = az.analyze(sample, conn=conn)
    print(f"  p_slow:       {result['p_slow']*100:.1f}%")
    print(f"  predicted ms: {result['predicted_ms']:.1f}")
    print(f"  plan cost:    {result['features']['plan_total_cost']:.0f}")
    print(f"  has_seq_scan: {result['features']['has_seq_scan']}")

    conn.close()