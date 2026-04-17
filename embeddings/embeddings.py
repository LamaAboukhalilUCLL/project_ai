import json
import os
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

HISTORY_FILE = "embeddings/history.json"

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, "r") as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

def store_fix(slow_query, fix, speedup_ms):
    history = load_history()
    embedding = model.encode(slow_query).tolist()
    entry = {
        "slow_query": slow_query,
        "fix": fix,
        "speedup_ms": speedup_ms,
        "embedding": embedding
    }
    history.append(entry)
    save_history(history)
    print(f"Stored fix. History now has {len(history)} entries.")

def find_similar(slow_query, top_k=1):
    history = load_history()
    if not history:
        return None

    query_embedding = model.encode(slow_query)

    best_score = -1
    best_entry = None

    for entry in history:
        stored_embedding = np.array(entry["embedding"])
        similarity = np.dot(query_embedding, stored_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
        )
        if similarity > best_score:
            best_score = similarity
            best_entry = entry

    if best_score > 0.7:
        print(f"Found similar past fix (similarity: {best_score:.2f})")
        return best_entry
    else:
        print(f"No similar fix found (best similarity: {best_score:.2f})")
        return None

if __name__ == "__main__":
    print("Testing embedding system...\n")

    print("Storing a test fix...")
    store_fix(
        slow_query="SELECT * FROM users WHERE location LIKE '%United States%'",
        fix="CREATE INDEX idx_users_location ON users(location); SELECT id, display_name FROM users WHERE location LIKE '%United States%' LIMIT 100;",
        speedup_ms=320.5
    )

    store_fix(
        slow_query="SELECT * FROM posts WHERE owner_user_id IN (SELECT id FROM users WHERE reputation > 1000)",
        fix="SELECT p.id, p.title FROM posts p JOIN users u ON p.owner_user_id = u.id WHERE u.reputation > 1000",
        speedup_ms=180.2
    )

    print("\nSearching for similar query...")
    result = find_similar("SELECT * FROM users WHERE location LIKE '%Germany%'")

    if result:
        print(f"\nSlow query: {result['slow_query']}")
        print(f"Suggested fix: {result['fix']}")
        print(f"Previous speedup: {result['speedup_ms']}ms")
    else:
        print("No similar fix found.")