from flask import Flask, render_template, jsonify
import json
import os

app = Flask(__name__)

HISTORY_FILE = "embeddings/history.json"
LOG_FILE = "dashboard/query_log.json"

def load_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        return json.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/queries")
def queries():
    return jsonify(load_log())

@app.route("/api/stats")
def stats():
    log = load_log()
    total = len(log)
    flagged = len([q for q in log if q.get("flagged")])
    speedups = [q["speedup_ms"] for q in log if q.get("speedup_ms") and q["speedup_ms"] > 0]
    avg_speedup = round(sum(speedups) / len(speedups), 1) if speedups else 0
    return jsonify({
        "total": total,
        "flagged": flagged,
        "avg_speedup": avg_speedup
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)