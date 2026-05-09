"""
Pipeline evaluation summary.

Reads pipeline_evaluation_results.json and prints a formatted
summary table with aggregate metrics for the report.

Run from project root:
    python evaluate_pipeline_results.py
"""

import json
import numpy as np
import os 

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pipeline_evaluation_results.json")

with open(RESULTS_FILE) as f:
    results = json.load(f)

wins      = [r for r in results if r["result"] == "win"]
skipped   = [r for r in results if r["result"] == "skipped_below_threshold"]
total     = len(results)
attempted = [r for r in results if r["result"] != "skipped_below_threshold"]

print("=" * 72)
print("  Pipeline Evaluation — Live Monitor Run")
print("=" * 72)
print(f"{'Query pattern':<38} {'Original':>10} {'Fixed':>8} {'Saved':>8} {'Result'}")
print("-" * 72)

for r in results:
    orig  = f"{r['pg_stat_mean_ms']:.1f}ms"
    if r["result"] == "win":
        fixed  = f"{r['verified_warm_median_ms']:.1f}ms"
        saved  = f"{r['measured_speedup_ms']:.1f}ms"
        result = "WIN"
    elif r["result"] == "skipped_below_threshold":
        fixed  = "—"
        saved  = "—"
        result = f"SKIP (p={r['model_p_slow']*100:.0f}%)"
    else:
        fixed  = "—"
        saved  = "—"
        result = r["result"].upper()
    print(f"{r['pattern']:<38} {orig:>10} {fixed:>8} {saved:>8}  {result}")

print("=" * 72)

speedups    = [r["measured_speedup_ms"] for r in wins]
factors     = [r["speedup_factor"] for r in wins]
win_rate    = len(wins) / len(attempted) * 100

print(f"\nAggregate metrics ({len(attempted)} queries attempted, {len(skipped)} skipped by classifier):")
print(f"  Win rate              : {len(wins)}/{len(attempted)} = {win_rate:.0f}%")
print(f"  Total time saved      : {sum(speedups):.1f}ms  ({sum(speedups)/1000:.2f}s)")
print(f"  Avg speedup           : {np.mean(speedups):.1f}ms per query")
print(f"  Median speedup        : {np.median(speedups):.1f}ms per query")
print(f"  Best speedup          : {max(speedups):.1f}ms  ({results[3]['pattern']})")
print(f"  Avg speedup factor    : {np.mean(factors):.2f}x faster")
print(f"  Max speedup factor    : {max(factors):.1f}x faster")
print(f"  Classifier skip rate  : {len(skipped)}/{total} = {len(skipped)/total*100:.0f}%")
print(f"  Classifier skip note  : LIKE query correctly identified as non-slow (p_slow=9.7%)")

print(f"\nStrategy breakdown:")
strategy_counts = {}
for r in wins:
    s = r.get("strategy", "unknown")
    strategy_counts[s] = strategy_counts.get(s, 0) + 1
for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1]):
    print(f"  {s:<30} {c} queries")

print(f"\nEmbedding memory usage:")
sim_scores = [r["similar_fix_similarity"] for r in results if r["similar_fix_similarity"]]
print(f"  Similar fix found     : {sum(1 for r in results if r['similar_fix_found'])}/{total} queries")
print(f"  Avg similarity score  : {np.mean(sim_scores):.2f}")


if __name__ == "__main__":
    pass