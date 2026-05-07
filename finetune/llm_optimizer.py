"""
Step 3 of fine-tuning pipeline: inference wrapper.

Provides suggest_optimized_query(slow_query) — a drop-in replacement for
the GPT API call in monitor.py.

Includes a SQL validator (_clean_sql) that fixes common model output errors
before candidates reach the database verification step.
"""

import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────── Config ───────────────────────────

FINETUNED_MODEL_DIR = "finetune/codet5-finetuned"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 160


# ─────────────────────────── SQL Validator ───────────────────────────

def _extract_aliases(sql: str) -> set:
    """Extract all table aliases defined in FROM/JOIN clauses."""
    aliases = set()
    patterns = [
        r'\bFROM\s+\w+\s+(\w+)',
        r'\bJOIN\s+\w+\s+(\w+)',
    ]
    keywords = {'ON', 'WHERE', 'AND', 'OR', 'AS', 'SET', 'ORDER',
                'GROUP', 'HAVING', 'LIMIT', 'OFFSET', 'INNER',
                'LEFT', 'RIGHT', 'OUTER', 'CROSS', 'NATURAL'}
    for pattern in patterns:
        for match in re.finditer(pattern, sql, re.IGNORECASE):
            alias = match.group(1).upper()
            if alias not in keywords:
                aliases.add(match.group(1))
    return aliases


def _clean_sql(sql: str, original_query: str) -> str | None:
    """
    Attempt to fix common model output errors.
    Returns cleaned SQL string, or None if the query is too broken to fix.

    Fixes applied:
    1. SELECT p.* with undefined alias -> SELECT *
    2. Undefined alias.column in SELECT clause -> remove alias prefix
    3. COUNT/AVG with too many arguments -> keep first argument only
    4. Broken JOIN ON clause -> discard
    """
    if not sql or not sql.strip():
        return None

    sql = sql.strip()

    if not sql.upper().lstrip().startswith("SELECT"):
        return None

    if "FROM" not in sql.upper():
        return None

    defined_aliases = _extract_aliases(sql)

    # Fix 1: SELECT p.* where p is not defined -> SELECT *
    def fix_undefined_star(match):
        alias = match.group(1)
        if alias not in defined_aliases:
            return "SELECT *"
        return match.group(0)

    sql = re.sub(r'\bSELECT\s+(\w+)\.\*', fix_undefined_star, sql, flags=re.IGNORECASE)
    # Fix: JOIN condition ending with corrupted u.id / p.id / v.id
    sql = re.sub(r'\b([upv])[.\-_\'!?](?:id|ie|ia|ied|iD|99|ird)\b', r'\1.id', sql, flags=re.IGNORECASE)    
    # Fix: p_owner_user_id -> p.owner_user_id (underscore instead of dot)
    sql = re.sub(r'\b([a-z])_owner_user_id\b', r'\1.owner_user_id', sql, flags=re.IGNORECASE)
    # Fix: GROUP BY owner -> GROUP BY u.id, u.display_name (too vague, just remove it)
    sql = re.sub(r'GROUP BY owner\b', 'GROUP BY u.id', sql, flags=re.IGNORECASE)
    # Fix: JOINS -> JOIN
    sql = re.sub(r'\bJOINS\b', 'JOIN', sql, flags=re.IGNORECASE)
    # Fix: GROUP BY owner_user... -> GROUP BY u.id
    sql = re.sub(r'GROUP BY owner[_\w]*', 'GROUP BY u.id', sql, flags=re.IGNORECASE)
    # Fix: GROUP BY u dis -> GROUP BY u.id, u.display_name  
    sql = re.sub(r'GROUP BY u dis\w*', 'GROUP BY u.id, u.display_name', sql, flags=re.IGNORECASE)
    # Fix: GROUP BY uvg... -> GROUP BY u.id
    sql = re.sub(r'GROUP BY u[vbfg]\w*', 'GROUP BY u.id', sql, flags=re.IGNORECASE)
    # Fix: Score DESC -> score DESC (case artifact)
    sql = sql.replace('Score DESC', 'score DESC')
    # Fix: owner_user/id -> owner_user_id
    sql = re.sub(r'owner_user[/\\]id', 'owner_user_id', sql, flags=re.IGNORECASE)
    # Fix: GROUP BY u.id followed by garbage -> GROUP BY u.id, u.display_name
    sql = re.sub(r'GROUP BY u\.id[\W_]\S*', 'GROUP BY u.id, u.display_name', sql, flags=re.IGNORECASE)
    # Fix: GROUP BY vote_count DESC -> remove GROUP BY entirely (invalid)
    sql = re.sub(r'GROUP BY \w+ DESC', '', sql, flags=re.IGNORECASE)

    # Fix 2: undefined alias.column in SELECT clause
    select_match = re.match(r'(SELECT\s+)(.*?)\s+FROM\b', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(2)
        fixed_select = select_clause
        alias_col_pattern = re.compile(r'\b(\w+)\.(\w+)\b')
        for match in alias_col_pattern.finditer(select_clause):
            alias = match.group(1)
            col = match.group(2)
            if alias not in defined_aliases and alias.upper() not in {'NEW', 'OLD'}:
                fixed_select = fixed_select.replace(match.group(0), col, 1)
        sql = sql[:select_match.start(2)] + fixed_select + sql[select_match.end(2):]

    # Fix 3: broken JOIN ON clause with no condition
    if re.search(r'\bON\s+\w+\s*$', sql, re.IGNORECASE):
        return None

    # Fix 4: COUNT with too many args
    def fix_count(match):
        args = match.group(1)
        parts = args.split(',')
        if len(parts) > 1:
            return f"COUNT({parts[0].strip()})"
        return match.group(0)
    sql = re.sub(r'\bCOUNT\(([^)]+)\)', fix_count, sql, flags=re.IGNORECASE)

    # Fix 5: AVG with too many args
    def fix_avg(match):
        args = match.group(1)
        parts = args.split(',')
        if len(parts) > 1:
            return f"AVG({parts[0].strip()})"
        return match.group(0)
    sql = re.sub(r'\bAVG\(([^)]+)\)', fix_avg, sql, flags=re.IGNORECASE)

    # Final validity check
    if not sql.upper().lstrip().startswith("SELECT"):
        return None
    if "FROM" not in sql.upper():
        return None

    return sql


# ─────────────────────────── Optimizer ───────────────────────────

class LLMOptimizer:
    """
    Wraps the fine-tuned T5 model for inference.
    Falls back to GPT-4o-mini if the model hasn't been trained yet.
    """

    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._device = None
        self._use_fallback = False

        if os.path.exists(FINETUNED_MODEL_DIR):
            self._load_finetuned()
        else:
            print(
                f"[LLMOptimizer] WARNING: Fine-tuned model not found at "
                f"'{FINETUNED_MODEL_DIR}'.\n"
                f"  Run 'python finetune/finetune.py' to train it.\n"
                f"  Falling back to GPT-4o-mini for now."
            )
            self._use_fallback = True

    def _load_finetuned(self):
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        print(f"[LLMOptimizer] Loading fine-tuned model from {FINETUNED_MODEL_DIR}...")
        self._tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL_DIR)
        self._model.to(self._device)
        self._model.eval()
        print(f"[LLMOptimizer] Model loaded on {self._device}.")

    def _generate_with_model(self, slow_query: str) -> str:
        source = "optimize sql: " + slow_query.strip()
        enc = self._tokenizer(
            source,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_length=MAX_TARGET_LEN,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _generate_with_gpt(self, slow_query: str, schema: str = "") -> str:
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = (
            f"Rewrite this slow PostgreSQL query to be faster.\n"
            f"Schema context: {schema if schema else 'posts, users, votes tables'}\n"
            f"Slow query: {slow_query}\n"
            f"Reply with ONLY the optimized SQL. No explanation."
        )
        response = gpt_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
        )
        return response.choices[0].message.content.strip()

    def suggest(self, slow_query: str, schema: str = "") -> dict:
        try:
            if self._use_fallback:
                optimized = self._generate_with_gpt(slow_query, schema)
                source = "fallback_gpt"
            else:
                optimized = self._generate_with_model(slow_query)
                source = "finetuned_model"
        except Exception as e:
            print(f"[LLMOptimizer] Generation error: {e}")
            return {"optimized_sql": slow_query, "source": "error", "valid": False}

        cleaned = _clean_sql(optimized, slow_query)
        if cleaned is None:
            cleaned = slow_query
            valid = False
        else:
            valid = True

        return {"optimized_sql": cleaned, "source": source, "valid": valid}

    def suggest_candidates(self, slow_query: str, schema: str = "", n: int = 3) -> list[dict]:
        """
        Generate n candidate optimizations using beam search + temperature sampling.
        All candidates are validated and cleaned before being returned.
        """
        if self._use_fallback:
            result = self.suggest(slow_query, schema)
            return [{**result, "strategy": "gpt_suggestion"}]

        source_text = "optimize sql: " + slow_query.strip()
        enc = self._tokenizer(
            source_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(self._device)

        candidates = []
        seen = set()

        # Candidate 1: beam search — most confident output
        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_length=MAX_TARGET_LEN,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        raw = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        print(f"    [BEAM RAW]     : {raw[:120]}")
        cleaned = _clean_sql(raw, slow_query)
        print(f"    [BEAM CLEANED] : {cleaned[:120] if cleaned else 'DISCARDED'}")
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            candidates.append({
                "optimized_sql": cleaned,
                "source": "finetuned_model",
                "valid": True,
                "strategy": _infer_strategy(slow_query, cleaned),
            })

        # Candidates 2 and 3: temperature sampling for variety
        for temp in [0.7, 1.0]:
            if len(candidates) >= n:
                break
            try:
                with torch.no_grad():
                    out = self._model.generate(
                        **enc,
                        max_length=MAX_TARGET_LEN,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.95,
                        no_repeat_ngram_size=3,
                    )
                raw = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
                print(f"    [TEMP {temp} RAW]     : {raw[:120]}")
                cleaned = _clean_sql(raw, slow_query)
                print(f"    [TEMP {temp} CLEANED] : {cleaned[:120] if cleaned else 'DISCARDED'}")
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    candidates.append({
                        "optimized_sql": cleaned,
                        "source": "finetuned_model",
                        "valid": True,
                        "strategy": _infer_strategy(slow_query, cleaned),
                    })
            except Exception:
                continue

        # Fallback if nothing valid generated
        if not candidates:
            result = self.suggest(slow_query, schema)
            if result["valid"]:
                candidates.append({
                    **result,
                    "strategy": _infer_strategy(slow_query, result["optimized_sql"])
                })

        return candidates


def _infer_strategy(original: str, optimized: str) -> str:
    orig_upper = original.upper()
    opt_upper = optimized.upper()
    if "LIMIT" in opt_upper and "LIMIT" not in orig_upper:
        return "add LIMIT"
    if "SELECT *" in orig_upper and "SELECT *" not in opt_upper:
        return "remove SELECT *"
    if opt_upper.count("JOIN") < orig_upper.count("JOIN"):
        return "reduce JOINs"
    if orig_upper.count("SELECT") > 1 and opt_upper.count("SELECT") <= orig_upper.count("SELECT"):
        return "rewrite subquery as JOIN"
    if "WHERE" in opt_upper and "WHERE" not in orig_upper:
        return "add WHERE filter"
    return "rewrite query"


# ─────────────────────────── CLI smoke test ───────────────────────────

if __name__ == "__main__":
    test_queries = [
        "SELECT * FROM posts ORDER BY score DESC",
        "SELECT * FROM users WHERE location LIKE '%United States%'",
        "SELECT * FROM posts p JOIN users u ON p.owner_user_id = u.id",
    ]

    print("Loading LLMOptimizer...")
    optimizer = LLMOptimizer()
    print()

    for q in test_queries:
        print(f"Input   : {q}")
        result = optimizer.suggest(q)
        print(f"Output  : {result['optimized_sql']}")
        print(f"Source  : {result['source']}  |  Valid: {result['valid']}")
        print()