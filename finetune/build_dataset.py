"""
Step 1 of fine-tuning pipeline: build the training dataset.

Reads every slow query from training_data.csv, sends each one to GPT-4o-mini
to get an optimized version, and saves the (slow, optimized) pairs to
finetune/finetune_pairs.json.

Run from the project root:
    python finetune/build_dataset.py

Expected output: finetune/finetune_pairs.json  (~161 pairs)
Cost: ~$0.02 with GPT-4o-mini at current pricing.
Runtime: ~3-5 minutes (API calls, rate-limited to avoid 429 errors).
"""

import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────── Config ───────────────────────────

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CSV_PATH = "training_data.csv"
OUTPUT_PATH = "finetune/finetune_pairs.json"

# The schema description we give GPT so it understands the tables.
# Keep it short — we only need the column names, not full types.
SCHEMA_CONTEXT = """
Database: stackexchange_db (Stats Stack Exchange dataset)
Tables:
  posts(id, post_type_id, accepted_answer_id, creation_date, score, view_count,
        body, owner_user_id, title, tags, answer_count, comment_count,
        favorite_count, closed_date, community_owned_date)
  users(id, reputation, creation_date, display_name, last_access_date,
        website_url, location, about_me, views, up_votes, down_votes)
  votes(id, post_id, vote_type_id, creation_date)

Indexes that exist: primary keys on id columns only.
""".strip()


def get_optimized_query(slow_query: str) -> str | None:
    """
    Ask GPT-4o-mini for one optimized version of slow_query.
    Returns the optimized SQL string, or None if the call fails.
    """
    prompt = f"""You are a PostgreSQL performance expert.

The following slow SQL query runs against this schema:
{SCHEMA_CONTEXT}

Slow query:
{slow_query}

Rewrite it as a single, complete, runnable SELECT that is faster.
Common strategies: add LIMIT, select specific columns instead of *, rewrite
correlated subqueries as JOINs, avoid functions in WHERE clauses, etc.

Reply with ONLY the optimized SQL. No explanation. No markdown. No semicolon."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        result = response.choices[0].message.content.strip()
        # Strip any accidental markdown fences GPT might add
        if result.startswith("```"):
            lines = result.split("\n")
            result = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()
        return result if result.upper().startswith("SELECT") else None
    except Exception as e:
        print(f"    [API error: {e}]")
        return None


def main():
    os.makedirs("finetune", exist_ok=True)

    # Load existing pairs so we can resume if interrupted
    existing_pairs = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            existing_pairs = json.load(f)
        print(f"Resuming — {len(existing_pairs)} pairs already saved.")

    existing_slow = {p["slow"] for p in existing_pairs}

    # Load slow queries from training data
    df = pd.read_csv(CSV_PATH)
    slow_queries = df[df["label"] == "slow"]["query_text"].tolist()
    remaining = [q for q in slow_queries if q not in existing_slow]

    print(f"Total slow queries : {len(slow_queries)}")
    print(f"Already processed  : {len(existing_pairs)}")
    print(f"Remaining          : {len(remaining)}")
    print()

    pairs = list(existing_pairs)
    failed = []

    for i, slow_query in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] {slow_query[:80]}{'...' if len(slow_query) > 80 else ''}")

        optimized = get_optimized_query(slow_query)

        if optimized:
            pairs.append({"slow": slow_query, "optimized": optimized})
            print(f"    → {optimized[:80]}{'...' if len(optimized) > 80 else ''}")
        else:
            failed.append(slow_query)
            print("    → FAILED (skipped)")

        # Save after every query so progress is never lost
        with open(OUTPUT_PATH, "w") as f:
            json.dump(pairs, f, indent=2)

        # Rate limiting: 0.5s between calls to stay well under OpenAI limits
        time.sleep(0.5)

    print()
    print("=" * 60)
    print(f"Done! Pairs saved   : {len(pairs)}")
    print(f"Failed              : {len(failed)}")
    print(f"Output file         : {OUTPUT_PATH}")

    if failed:
        print("\nFailed queries (you can retry manually):")
        for q in failed:
            print(f"  - {q}")


if __name__ == "__main__":
    main()