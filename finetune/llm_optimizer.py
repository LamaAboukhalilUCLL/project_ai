"""
LLM optimizer using fine-tuned sqlcoder-2b via QLoRA adapter.
Falls back to GPT-4o-mini if the fine-tuned model is not found.
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

FINETUNED_MODEL_DIR = "finetune/sqlcoder-finetuned"
BASE_MODEL_NAME     = "defog/sqlcoder-7b"
MAX_NEW_TOKENS      = 128

SCHEMA = (
    "CREATE TABLE posts (id INT, post_type_id INT, score INT, view_count INT, "
    "owner_user_id INT, title TEXT, answer_count INT, creation_date TIMESTAMP);\n"
    "CREATE TABLE users (id INT, display_name TEXT, reputation INT, location TEXT, "
    "views INT, creation_date TIMESTAMP);\n"
    "CREATE TABLE votes (id INT, post_id INT, vote_type_id INT, creation_date TIMESTAMP);"
)


def _make_prompt(slow_query):
    return (
        f"### Task\nRewrite the following slow PostgreSQL query to be faster.\n\n"
        f"### Database Schema\n{SCHEMA}\n\n"
        f"### Slow Query\n{slow_query.strip()}\n\n"
        f"### Optimized Query\n"
    )


def _extract_sql(text):
    if "### Optimized Query" in text:
        text = text.split("### Optimized Query")[-1]
    text = re.sub(r"```sql|```", "", text, flags=re.IGNORECASE)
    lines = []
    for line in text.strip().splitlines():
        lines.append(line)
        if line.strip().endswith(";"):
            break
    sql = "\n".join(lines).strip().rstrip(";").strip()
    # Remove OFFSET 0 artifacts
    sql = re.sub(r'\s+OFFSET\s+0\b.*$', '', sql, flags=re.IGNORECASE | re.DOTALL).strip()
    return sql if sql.upper().startswith("SELECT") else None


def _clean_sql(sql, original_query):
    """Basic validator — same as before but simplified since sqlcoder is cleaner."""
    if not sql or not sql.strip():
        return None
    sql = sql.strip()
    if not sql.upper().lstrip().startswith("SELECT"):
        return None
    if "FROM" not in sql.upper():
        return None
    # Fix JOINS -> JOIN
    sql = re.sub(r'\bJOINS\b', 'JOIN', sql, flags=re.IGNORECASE)
    return sql


class LLMOptimizer:
    def __init__(self):
        self._model     = None
        self._tokenizer = None
        self._device    = None
        self._use_fallback = False

        if os.path.exists(FINETUNED_MODEL_DIR):
            self._load_finetuned()
        else:
            print(
                f"[LLMOptimizer] Fine-tuned model not found at '{FINETUNED_MODEL_DIR}'.\n"
                f"  Run 'python finetune/finetune_sqlcoder.py' to train it.\n"
                f"  Falling back to GPT-4o-mini."
            )
            self._use_fallback = True

    def _load_finetuned(self):
        print(f"[LLMOptimizer] Loading fine-tuned sqlcoder from {FINETUNED_MODEL_DIR}...")
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL_DIR, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_DIR)
        self._model.eval()
        print(f"[LLMOptimizer] Model loaded on {self._device}.")

    def _generate(self, slow_query, temperature=None):
        prompt = _make_prompt(slow_query)
        enc = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            if temperature:
                out = self._model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.95,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            else:
                out = self._model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=4,
                    pad_token_id=self._tokenizer.eos_token_id,
                    repetition_penalty=1.3,
            )   

        full = self._tokenizer.decode(out[0], skip_special_tokens=True)
        return _extract_sql(full)

    def _generate_with_gpt(self, slow_query, schema=""):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = (
            f"Rewrite this slow PostgreSQL query to be faster.\n"
            f"Schema: posts(id,score,view_count,owner_user_id,title,post_type_id,answer_count) "
            f"users(id,display_name,reputation,location) votes(id,post_id,vote_type_id)\n"
            f"Slow query: {slow_query}\n"
            f"Reply with ONLY the optimized SQL."
        )
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=300,
        )
        return r.choices[0].message.content.strip()

    def suggest(self, slow_query, schema=""):
        try:
            if self._use_fallback:
                raw = self._generate_with_gpt(slow_query, schema)
                source = "fallback_gpt"
            else:
                raw = self._generate(slow_query)
                source = "sqlcoder_finetuned"
        except Exception as e:
            print(f"[LLMOptimizer] Error: {e}")
            return {"optimized_sql": slow_query, "source": "error", "valid": False}

        cleaned = _clean_sql(raw, slow_query) if raw else None
        if cleaned is None:
            return {"optimized_sql": slow_query, "source": source, "valid": False}
        return {"optimized_sql": cleaned, "source": source, "valid": True}

    def suggest_candidates(self, slow_query, schema="", n=3):
        if self._use_fallback:
            result = self.suggest(slow_query, schema)
            return [{**result, "strategy": "gpt_suggestion"}]

        candidates = []
        seen = set()

        # Candidate 1: greedy beam search
        raw = self._generate(slow_query)
        if raw:
            cleaned = _clean_sql(raw, slow_query)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                candidates.append({
                    "optimized_sql": cleaned,
                    "source": "sqlcoder_finetuned",
                    "valid": True,
                    "strategy": _infer_strategy(slow_query, cleaned),
                })

        # Candidates 2-3: temperature sampling
        for temp in [0.7, 1.0]:
            if len(candidates) >= n:
                break
            try:
                raw = self._generate(slow_query, temperature=temp)
                if raw:
                    cleaned = _clean_sql(raw, slow_query)
                    if cleaned and cleaned not in seen:
                        seen.add(cleaned)
                        candidates.append({
                            "optimized_sql": cleaned,
                            "source": "sqlcoder_finetuned",
                            "valid": True,
                            "strategy": _infer_strategy(slow_query, cleaned),
                        })
            except Exception:
                continue

        if not candidates:
            result = self.suggest(slow_query, schema)
            if result["valid"]:
                candidates.append({**result, "strategy": _infer_strategy(slow_query, result["optimized_sql"])})

        return candidates


def _infer_strategy(original, optimized):
    orig = original.upper()
    opt  = optimized.upper()
    # Check most specific patterns first
    if "SELECT *" in orig and "SELECT *" not in opt:
        return "remove SELECT *"
    if orig.count("SELECT") > 1 and "JOIN" in opt and "GROUP BY" in opt:
        return "rewrite subquery as JOIN"
    if orig.count("SELECT") > 1 and opt.count("SELECT") <= orig.count("SELECT"):
        return "rewrite subquery as JOIN"
    if opt.count("JOIN") < orig.count("JOIN"):
        return "reduce JOINs"
    if "WHERE" in opt and "WHERE" not in orig:
        return "add WHERE filter"
    if "LIMIT" in opt and "LIMIT" not in orig:
        return "add LIMIT"
    return "rewrite query"


if __name__ == "__main__":
    queries = [
        "SELECT * FROM posts ORDER BY score DESC",
        "SELECT * FROM users WHERE location LIKE '%United States%'",
        "SELECT display_name, (SELECT COUNT(*) FROM posts WHERE owner_user_id = users.id) FROM users",
    ]
    print("Loading LLMOptimizer...")
    opt = LLMOptimizer()
    for q in queries:
        print(f"\nInput : {q}")
        r = opt.suggest(q)
        print(f"Output: {r['optimized_sql']}")
        print(f"Valid : {r['valid']}")