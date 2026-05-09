"""
Manual calibration check.

Picks N queries from the training set, runs each through the analyzer,
and compares predicted execution time against the recorded warm median.
Prints a side-by-side table and saves a scatter plot to evaluation.png.

This is the "spot check" version — it doesn't replace the held-out test
set evaluation in the notebook, but lets us eyeball calibration on
specific queries (e.g. queries near the boundary, very slow queries, etc.)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import random
import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from classifier.inference import QueryAnalyzer

DB_CONFIG = {
    "dbname": "stackexchange_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost",
    "port": "5432",
}

N_QUERIES = 20
RANDOM_SEED = 7  # change this to sample different queries
random.seed(RANDOM_SEED)


def main():
    print("Loading data...")
    df = pd.read_csv(os.path.join(os.path.dirname(SCRIPT_DIR), "training_data.csv"))

    print(f"  {len(df)} queries available")

    # Sample with stratification: half slow, half fast, so we see both regimes
    slow = df[df["label"] == "slow"].sample(n=min(N_QUERIES // 2, (df["label"] == "slow").sum()),
                                             random_state=RANDOM_SEED)
    fast = df[df["label"] == "fast"].sample(n=N_QUERIES - len(slow),
                                             random_state=RANDOM_SEED)
    sample = pd.concat([slow, fast]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    print(f"  sampled {len(sample)} queries ({len(slow)} slow, {len(fast)} fast)")

    print("\nLoading analyzer...")
    analyzer = QueryAnalyzer()

    print("Connecting to DB...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True

    print("\n" + "=" * 90)
    print(f"{'#':<3} {'label':<6} {'actual_ms':>11} {'predicted_ms':>14} "
          f"{'error_ms':>10} {'p_slow':>8} {'verdict':<10}")
    print("=" * 90)

    rows = []
    for i, row in sample.iterrows():
        query = row["query_text"]
        actual = row["execution_time_warm_ms"]
        actual_label = row["label"]

        result = analyzer.analyze(query, conn=conn)
        predicted = result["predicted_ms"]
        p_slow = result["p_slow"]

        # Verdict: did the classification head agree with the label?
        predicted_label = "slow" if p_slow > 0.5 else "fast"
        verdict = "OK" if predicted_label == actual_label else "MISS"

        # Error in ms (signed) and ratio (factor)
        error_ms = predicted - actual

        print(f"{i+1:<3} {actual_label:<6} {actual:>11.1f} {predicted:>14.1f} "
              f"{error_ms:>+10.1f} {p_slow:>7.1%} {verdict:<10}")

        rows.append({
            "query": query[:120],
            "actual_ms": actual,
            "predicted_ms": predicted,
            "p_slow": p_slow,
            "actual_label": actual_label,
            "predicted_label": predicted_label,
            "error_ms": error_ms,
        })

    conn.close()

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(SCRIPT_DIR, "evaluation_sample.csv"), index=False)
    print(f"\nSaved details to evaluation_sample.csv")

    # Summary metrics
    correct_clf = (out["actual_label"] == out["predicted_label"]).sum()
    print(f"\nClassification: {correct_clf}/{len(out)} correct "
          f"({correct_clf / len(out) * 100:.0f}%)")

    # Regression: log-space MAE on this sample
    log_actual = np.log1p(out["actual_ms"])
    log_pred = np.log1p(out["predicted_ms"])
    log_mae = np.abs(log_actual - log_pred).mean()
    print(f"Regression log-MAE: {log_mae:.3f}  "
          f"(predictions off by avg factor of {np.exp(log_mae):.2f}x)")

    # Visual: predicted vs actual
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#f05252" if l == "slow" else "#4a9eda"
              for l in out["actual_label"]]
    ax.scatter(out["actual_ms"], out["predicted_ms"], c=colors, alpha=0.7, s=80)
    lim = [0.05, max(out["actual_ms"].max(), out["predicted_ms"].max()) * 1.5]
    ax.plot(lim, lim, "k--", lw=1, label="Perfect prediction")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Actual warm execution time (ms)")
    ax.set_ylabel("Predicted execution time (ms)")
    ax.set_title(f"Calibration on {len(out)} sampled queries")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "evaluation.png"), dpi=120)

    print("Saved scatter plot to evaluation.png")


if __name__ == "__main__":
    main()