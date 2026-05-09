"""
Fine-tune defog/sqlcoder-2b on SQL optimization pairs using QLoRA.

Uses 4-bit quantization + LoRA so the 2B model fits in 8GB VRAM.
Produces a merged model saved to finetune/sqlcoder-finetuned/

Run from project root with venv312 activated:
    python finetune/finetune_sqlcoder.py

Expected runtime on RTX 5070 Laptop: ~15-25 minutes
"""

import json
import os
import random
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ─────────────────────────── Config ───────────────────────────

MODEL_NAME   = "defog/sqlcoder-7b"
PAIRS_PATH   = "finetune/finetune_pairs.json"
OUTPUT_DIR   = "finetune/sqlcoder-finetuned"
RESULTS_PATH = "finetune/finetune_results_sqlcoder.json"

EPOCHS        = 3       # QLoRA converges fast — 3 epochs is enough
BATCH_SIZE    = 2       # keep low for 8GB VRAM
GRAD_ACCUM    = 4       # effective batch size = 8
LEARNING_RATE = 2e-4
MAX_SEQ_LEN   = 512     # sqlcoder handles longer sequences
VAL_SPLIT     = 0.1
RANDOM_SEED   = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ─────────────────────────── Load pairs ───────────────────────────

print(f"Loading pairs from {PAIRS_PATH}...")
with open(PAIRS_PATH) as f:
    pairs = json.load(f)
print(f"  {len(pairs)} pairs loaded")

random.shuffle(pairs)
split = int(len(pairs) * (1 - VAL_SPLIT))
train_pairs = pairs[:split]
val_pairs   = pairs[split:]
print(f"  Train: {len(train_pairs)}  Val: {len(val_pairs)}")

# ─────────────────────────── Prompt format ────────────────────────────
# sqlcoder uses a specific instruction format

SCHEMA = (
    "CREATE TABLE posts (id INT, post_type_id INT, score INT, view_count INT, "
    "owner_user_id INT, title TEXT, answer_count INT, creation_date TIMESTAMP);\n"
    "CREATE TABLE users (id INT, display_name TEXT, reputation INT, location TEXT, "
    "views INT, creation_date TIMESTAMP);\n"
    "CREATE TABLE votes (id INT, post_id INT, vote_type_id INT, creation_date TIMESTAMP);"
)

def make_prompt(slow_query, optimized_query=None):
    prompt = (
        f"### Task\nRewrite the following slow PostgreSQL query to be faster.\n\n"
        f"### Database Schema\n{SCHEMA}\n\n"
        f"### Slow Query\n{slow_query.strip()}\n\n"
        f"### Optimized Query\n"
    )
    if optimized_query is not None:
        prompt += optimized_query.strip()
    return prompt


# ─────────────────────────── Tokenizer ───────────────────────────

print(f"\nLoading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("  Tokenizer loaded")


# ─────────────────────────── Tokenize ───────────────────────────

def tokenize(pair):
    full_prompt = make_prompt(pair["slow"], pair["optimized"])
    prompt_only  = make_prompt(pair["slow"])

    full_enc   = tokenizer(full_prompt,  max_length=MAX_SEQ_LEN, truncation=True)
    prompt_enc = tokenizer(prompt_only,  max_length=MAX_SEQ_LEN, truncation=True)

    input_ids      = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # Mask prompt tokens in labels so loss is only computed on the answer
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

print("\nTokenizing...")
train_dataset = Dataset.from_list([tokenize(p) for p in train_pairs])
val_dataset   = Dataset.from_list([tokenize(p) for p in val_pairs])
print(f"  Train: {len(train_dataset)}  Val: {len(val_dataset)}")


# ─────────────────────────── Model (4-bit QLoRA) ──────────────────────

print(f"\nLoading {MODEL_NAME} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# LoRA config — target attention layers
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ─────────────────────────── Training ────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="none",
    dataloader_num_workers=0,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

print(f"\nStarting QLoRA fine-tuning for {EPOCHS} epochs...")
trainer.train()

# ─────────────────────────── Save ────────────────────────────────────

print(f"\nSaving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Save results
history = []
for log in trainer.state.log_history:
    if "eval_loss" in log:
        history.append({
            "epoch": log.get("epoch"),
            "eval_loss": log.get("eval_loss"),
            "train_loss": log.get("loss"),
        })

results = {
    "model": MODEL_NAME,
    "epochs": EPOCHS,
    "train_pairs": len(train_pairs),
    "val_pairs": len(val_pairs),
    "history": history,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)

print(f"Done. Results saved to {RESULTS_PATH}")
print(f"\nNext step: update llm_optimizer.py to use sqlcoder-finetuned/")