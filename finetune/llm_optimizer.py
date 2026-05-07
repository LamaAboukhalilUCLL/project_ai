"""
Step 3 of fine-tuning pipeline: inference wrapper.

Provides suggest_optimized_query(slow_query) — a drop-in replacement for
the GPT API call in monitor.py.

Usage:
    from finetune.llm_optimizer import LLMOptimizer
    optimizer = LLMOptimizer()
    result = optimizer.suggest(slow_query)
    print(result["optimized_sql"])   # the rewritten query
    print(result["source"])          # "finetuned_model" or "fallback_gpt"

The class loads the fine-tuned model once and keeps it in memory for the
lifetime of the monitor process — no cold-load penalty per query.

Fallback behaviour: if the fine-tuned model is not found (i.e. you haven't
run finetune.py yet), the class falls back to GPT-4o-mini automatically so
the monitor continues to work. A warning is printed.
"""

import os
import re

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────── Config ───────────────────────────

FINETUNED_MODEL_DIR = "finetune/codet5-finetuned"
MAX_INPUT_LEN = 128
MAX_TARGET_LEN = 128


class LLMOptimizer:
    """
    Wraps the fine-tuned CodeT5 model for inference.
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
        """Load the fine-tuned CodeT5 model from disk."""
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        print(f"[LLMOptimizer] Loading fine-tuned model from {FINETUNED_MODEL_DIR}...")
        self._tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR, use_fast=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL_DIR)
        self._model.to(self._device)
        self._model.eval()
        print(f"[LLMOptimizer] Model loaded on {self._device}.")

    def _generate_with_model(self, slow_query: str) -> str:
        """Run inference with the fine-tuned model."""
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
                num_beams=4,        # beam search for better quality
                early_stopping=True,
                no_repeat_ngram_size=3,  # avoid repetitive output
            )

        result = self._tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return result

    def _generate_with_gpt(self, slow_query: str, schema: str = "") -> str:
        """GPT-4o-mini fallback — only used if fine-tuned model is missing."""
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
        """
        Generate one optimized version of slow_query.

        Returns:
            {
                "optimized_sql": str,   # the rewritten query
                "source": str,          # "finetuned_model" or "fallback_gpt"
                "valid": bool,          # True if output looks like a SELECT
            }
        """
        try:
            if self._use_fallback:
                optimized = self._generate_with_gpt(slow_query, schema)
                source = "fallback_gpt"
            else:
                optimized = self._generate_with_model(slow_query)
                source = "finetuned_model"
        except Exception as e:
            print(f"[LLMOptimizer] Generation error: {e}")
            return {
                "optimized_sql": slow_query,  # return original on failure
                "source": "error",
                "valid": False,
            }

        # Basic validity check — output must start with SELECT
        valid = bool(optimized) and optimized.upper().lstrip().startswith("SELECT")

        return {
            "optimized_sql": optimized,
            "source": source,
            "valid": valid,
        }

    def suggest_candidates(self, slow_query: str, schema: str = "", n: int = 3) -> list[dict]:
        """
        Generate n candidate optimizations using beam search with diverse beams.
        Returns a list of suggestion dicts sorted by uniqueness.
        This is the method called by monitor.py to get multiple candidates.

        Returns:
            List of {"optimized_sql": str, "source": str, "valid": bool, "strategy": str}
        """
        if self._use_fallback:
            # GPT fallback: call once and wrap in list
            result = self.suggest(slow_query, schema)
            return [{**result, "strategy": "gpt_suggestion"}]

        source_text = "optimize sql: " + slow_query.strip()
        enc = self._tokenizer(
            source_text,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(self._device)

        with torch.no_grad():
            # num_return_sequences gives us n distinct candidates
            # num_beam_groups enables diverse beam search
            out = self._model.generate(
                **enc,
                max_length=MAX_TARGET_LEN,
                num_beams=max(n * 2, 6),          # search wider than we return
                num_return_sequences=n,
                num_beam_groups=n,                 # diverse beam search
                diversity_penalty=0.5,             # push beams apart
                early_stopping=True,
                no_repeat_ngram_size=3,
            )

        candidates = []
        seen = set()
        for beam_output in out:
            sql = self._tokenizer.decode(beam_output, skip_special_tokens=True).strip()
            if sql in seen:
                continue
            seen.add(sql)
            valid = bool(sql) and sql.upper().lstrip().startswith("SELECT")
            # Infer a rough strategy label from the output
            strategy = _infer_strategy(slow_query, sql)
            candidates.append({
                "optimized_sql": sql,
                "source": "finetuned_model",
                "valid": valid,
                "strategy": strategy,
            })

        return candidates if candidates else [self.suggest(slow_query, schema)]


def _infer_strategy(original: str, optimized: str) -> str:
    """
    Guess a short strategy label by comparing original and optimized.
    This is used only for the dashboard display — it doesn't affect correctness.
    """
    orig_upper = original.upper()
    opt_upper = optimized.upper()

    if "LIMIT" in opt_upper and "LIMIT" not in orig_upper:
        return "add LIMIT"
    if "SELECT *" in orig_upper and "SELECT *" not in opt_upper:
        return "remove SELECT *"
    if opt_upper.count("JOIN") < orig_upper.count("JOIN"):
        return "reduce JOINs"
    if "SUBQUERY" in orig_upper or orig_upper.count("SELECT") > 1:
        if opt_upper.count("SELECT") <= orig_upper.count("SELECT"):
            return "rewrite subquery as JOIN"
    if "WHERE" in opt_upper and "WHERE" not in orig_upper:
        return "add WHERE filter"
    if "INDEX" in opt_upper:
        return "use index hint"
    return "rewrite query"


# ─────────────────────────── CLI smoke test ───────────────────────────

if __name__ == "__main__":
    """
    Quick sanity check. Run from the project root:
        python -m finetune.llm_optimizer
    """
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