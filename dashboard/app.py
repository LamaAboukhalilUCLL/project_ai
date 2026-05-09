from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

LOG_FILE = "dashboard/query_log.json"


def load_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def normalize_entry(q):
    """
    Normalize a log entry to a consistent format regardless of
    whether it came from the old GPT pipeline or the new sqlcoder pipeline.
    """
    if "pg_stat_mean_ms" in q:
        model = q.get("model", {})
        p_slow = model.get("p_slow", 0)
        candidates = q.get("candidates_ranked", [])
        top = candidates[0] if candidates else {}
        verified = q.get("top_candidate_verified")
        speedup = q.get("measured_speedup_ms")
        return {
            "query": q.get("query", ""),
            "avg_time_ms": q.get("pg_stat_mean_ms", 0),
            "confidence": round(p_slow * 100, 1),
            "reason": q.get("reason", ""),
            "fix": top.get("sql", "") if top else "",
            "strategy": top.get("strategy", ""),
            "speedup_ms": round(speedup, 1) if speedup and speedup > 0 else 0,
            "measured": bool(verified),
            "warm_median_ms": verified.get("warm_median_ms") if verified else None,
            "flagged": p_slow >= 0.5,
            "timestamp": q.get("timestamp", ""),
            "llm_source": q.get("llm_source", "sqlcoder"),
            "type": q.get("type", "slow_query_handled"),
        }
    # Old format
    return {
        "query": q.get("query", ""),
        "avg_time_ms": q.get("avg_time_ms", 0),
        "confidence": q.get("confidence", 0),
        "reason": q.get("reason", ""),
        "fix": q.get("fix", ""),
        "strategy": "",
        "speedup_ms": q.get("speedup_ms", 0) or 0,
        "measured": q.get("speedup_ms", 0) > 0,
        "warm_median_ms": None,
        "flagged": q.get("flagged", False),
        "timestamp": q.get("timestamp", ""),
        "llm_source": "gpt",
        "type": "slow_query_handled",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/queries")
def queries():
    log = load_log()
    handled = [
        normalize_entry(q) for q in log
        if q.get("type", "slow_query_handled") == "slow_query_handled"
    ]
    return jsonify(handled)


@app.route("/api/stats")
def stats():
    log = load_log()
    handled = [
        normalize_entry(q) for q in log
        if q.get("type", "slow_query_handled") == "slow_query_handled"
    ]
    total = len(handled)
    flagged = sum(1 for q in handled if q["flagged"])
    fixed = sum(1 for q in handled if q["speedup_ms"] > 0)
    speedups = [q["speedup_ms"] for q in handled if q["speedup_ms"] > 0]
    avg_speedup = round(sum(speedups) / len(speedups), 1) if speedups else 0
    total_saved = round(sum(speedups), 0) if speedups else 0
    return jsonify({
        "total": total,
        "flagged": flagged,
        "fixed": fixed,
        "avg_speedup": avg_speedup,
        "total_saved_ms": total_saved,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)