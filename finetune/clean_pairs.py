import json
import re

PAIRS_PATH = "finetune/finetune_pairs.json"

with open(PAIRS_PATH) as f:
    pairs = json.load(f)

print(f"Total pairs before: {len(pairs)}")

def clean_sql(sql):
    # Remove HTML entities like &acute; &ale; &quot; etc.
    sql = re.sub(r'&[a-zA-Z]+;', '', sql)
    # Fix tokenizer artifacts where 'id' gets decoded as '99'
    sql = sql.replace('u.99', 'u.id')
    sql = sql.replace('p.99', 'p.id')
    sql = sql.replace('v.99', 'v.id')
    # Remove any remaining non-ASCII characters
    sql = sql.encode('ascii', 'ignore').decode('ascii')
    return sql.strip()

cleaned = []
fixed_count = 0
removed_count = 0

for pair in pairs:
    cleaned_opt = clean_sql(pair["optimized"])
    cleaned_slow = clean_sql(pair["slow"])

    # Track what changed
    if cleaned_opt != pair["optimized"] or cleaned_slow != pair["slow"]:
        fixed_count += 1

    # Only keep pair if optimized still looks like valid SQL after cleaning
    if cleaned_opt.upper().startswith("SELECT") and "FROM" in cleaned_opt.upper():
        cleaned.append({
            "slow": cleaned_slow,
            "optimized": cleaned_opt
        })
    else:
        removed_count += 1
        print(f"  Removed broken pair: {cleaned_opt[:80]}")

with open(PAIRS_PATH, "w") as f:
    json.dump(cleaned, f, indent=2)

print(f"Fixed  : {fixed_count} pairs")
print(f"Removed: {removed_count} pairs (too broken after cleaning)")
print(f"Total pairs after: {len(cleaned)}")
print("Saved to finetune/finetune_pairs.json")