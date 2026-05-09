"""
Step 2 of fine-tuning pipeline: fine-tune CodeT5-small.

Loads finetune/finetune_pairs.json, trains Salesforce/codet5-small on the
(slow_query → optimized_query) pairs, and saves the fine-tuned model to
finetune/codet5-finetuned/.

Run from the project root:
    python finetune/finetune.py

Expected runtime:
  - With NVIDIA GPU (your setup): ~5-15 minutes
  - Without GPU (CPU only):       ~45-90 minutes

Expected output:
  finetune/codet5-finetuned/          ← fine-tuned model weights
  finetune/finetune_results.json      ← training loss history per epoch
"""

import json
import os
import random

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─────────────────────────── Config ───────────────────────────

MODEL_NAME = "t5-small"   # 60M params, code-aware encoder-decoder
PAIRS_PATH = "finetune/finetune_pairs.json"
OUTPUT_DIR = "finetune/codet5-finetuned"
RESULTS_PATH = "finetune/finetune_results.json"

# Training hyperparameters — tuned for ~150 pairs on a gaming laptop GPU
EPOCHS = 20           # MORE LEARNINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG
BATCH_SIZE = 8        # fits comfortably in 4GB+ VRAM; reduce to 4 if you get OOM
LEARNING_RATE = 5e-4  # standard for seq2seq fine-tuning on small datasets
MAX_INPUT_LEN = 128   # SQL queries are short; 128 tokens is plenty
MAX_TARGET_LEN = 128  # optimized queries are similar length
VAL_SPLIT = 0.1       # 10% held out for validation
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ─────────────────────────── Dataset ───────────────────────────

class SQLPairDataset(Dataset):
    """
    Simple dataset that holds (input_ids, attention_mask, labels) tensors.
    The 'labels' are the tokenized optimized queries. The -100 padding trick
    tells the model to ignore padding positions when computing loss.
    """
    def __init__(self, pairs, tokenizer):
        self.data = []
        for pair in pairs:
            # Prefix helps the model understand the task (T5-style)
            source = "optimize sql: " + pair["slow"].strip()
            target = pair["optimized"].strip()

            enc = tokenizer(
                source,
                max_length=MAX_INPUT_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            dec = tokenizer(
                target,
                max_length=MAX_TARGET_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
                

            # Replace padding token id in labels with -100 so loss ignores them
            labels = dec["input_ids"].squeeze()
            labels[labels == tokenizer.pad_token_id] = -100

            self.data.append({
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": labels,
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ─────────────────────────── Training ───────────────────────────

def compute_val_loss(model, val_loader, device):
    """Run one pass over the validation set and return average loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
    model.train()
    return total_loss / len(val_loader)


def main():
    # ── Detect device ──
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU found — using CPU (this will be slower)")

    # ── Load pairs ──
    if not os.path.exists(PAIRS_PATH):
        print(f"ERROR: {PAIRS_PATH} not found.")
        print("Run 'python finetune/build_dataset.py' first.")
        return

    with open(PAIRS_PATH, "r") as f:
        pairs = json.load(f)

    print(f"Loaded {len(pairs)} training pairs from {PAIRS_PATH}")

    if len(pairs) < 10:
        print("ERROR: Too few pairs to fine-tune. Run build_dataset.py first.")
        return

    # ── Train/val split ──
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * VAL_SPLIT))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]
    print(f"Train: {len(train_pairs)} pairs   Val: {len(val_pairs)} pairs")

    # ── Load model and tokenizer ──
    print(f"\nLoading {MODEL_NAME} ...")
    print("(First run will download ~230MB — this is normal.)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.to(device)
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Build datasets and loaders ──
    print("\nTokenizing dataset...")
    train_dataset = SQLPairDataset(train_pairs, tokenizer)
    val_dataset = SQLPairDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # ── Training loop ──
    print(f"\nStarting fine-tuning for {EPOCHS} epochs...\n")
    print(f"{'Epoch':<7} {'Train Loss':<14} {'Val Loss':<12}")
    print("-" * 35)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = compute_val_loss(model, val_loader, device)

        print(f"{epoch:<7} {avg_train_loss:<14.4f} {avg_val_loss:<12.4f}", end="")

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(" ← best, saved", end="")

        print()  # newline

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
        })

    # ── Save training results ──
    results = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "best_epoch": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # ── Quick smoke test ──
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best model: epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"Saved to  : {OUTPUT_DIR}/")
    print(f"Results   : {RESULTS_PATH}")
    print()

    print("Quick smoke test on 3 training examples:")
    model.eval()
    for pair in random.sample(train_pairs, min(3, len(train_pairs))):
        source = "optimize sql: " + pair["slow"].strip()
        enc = tokenizer(
            source,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_length=MAX_TARGET_LEN,
                num_beams=4,
                early_stopping=True,
            )

        generated = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n  Input   : {pair['slow'][:90]}")
        print(f"  Expected: {pair['optimized'][:90]}")
        print(f"  Model   : {generated[:90]}")

    print()
    print("Fine-tuning complete. Next step:")
    print("  The model is saved and ready.")
    print("  monitor.py already calls llm_optimizer.py — no further changes needed.")


if __name__ == "__main__":
    main()