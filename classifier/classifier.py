"""
Multi-task classifier for SQL query analysis.

Two output heads sharing a feature backbone:
  - Classification: P(slow), trained with BCE
  - Regression: predicted log-execution-time in ms, trained with MSE

Joint loss = alpha * BCE + (1 - alpha) * QuantileLoss(tau=0.7).
(Originally MSE; switched to quantile loss to reduce systematic
under-prediction of slow queries — see notebook section 7 for
the full rationale.)

Inputs: 8 text-derived features + 12 plan-derived features = 20 total.
Target labels are derived from training_data.csv:
  - slow/fast label is taken directly
  - log execution time is computed from execution_time_warm_ms
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, r2_score,
)
import pickle
import json
import os

# ───────────────────────── Config ─────────────────────────

CSV_PATH = "training_data.csv"
MODEL_PATH = "classifier/query_classifier.pth"
SCALER_PATH = "classifier/scaler.pkl"
META_PATH = "classifier/model_meta.json"

EPOCHS = 150
BATCH_SIZE = 32
LR = 1e-3
ALPHA = 0.6        # weight on classification loss vs regression loss
DROPOUT = 0.3
RANDOM_SEED = 42

FEATURE_COLS = [
    # Text-derived
    "has_select_star", "has_like_wildcard", "join_count",
    "has_subquery", "has_group_by", "has_order_by_no_limit",
    "has_or", "has_function_in_where",
    # Plan-derived
    "plan_total_cost", "plan_rows", "actual_rows", "plan_depth",
    "shared_hit", "shared_read",
    "has_seq_scan", "has_index_scan", "has_bitmap_scan",
    "has_hash_join", "has_nested_loop", "has_merge_join",
]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ───────────────────────── Data ─────────────────────────

print("Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} queries loaded")
print(f"  slow={int((df['label']=='slow').sum())}  "
      f"fast={int((df['label']=='fast').sum())}")

X = df[FEATURE_COLS].values.astype(np.float32)
y_clf = (df["label"] == "slow").astype(np.float32).values
# Use log1p to compress the wide time range (sub-ms to 10s).
# log1p handles zeros gracefully.
y_reg = np.log1p(df["execution_time_warm_ms"].astype(np.float32).values)

X_train, X_test, y_clf_train, y_clf_test, y_reg_train, y_reg_test = train_test_split(
    X, y_clf, y_reg,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=y_clf,
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_clf_train_t = torch.tensor(y_clf_train, dtype=torch.float32).unsqueeze(1)
y_clf_test_t  = torch.tensor(y_clf_test,  dtype=torch.float32).unsqueeze(1)
y_reg_train_t = torch.tensor(y_reg_train, dtype=torch.float32).unsqueeze(1)
y_reg_test_t  = torch.tensor(y_reg_test,  dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_clf_train_t, y_reg_train_t)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ───────────────────────── Model ─────────────────────────

class MultiTaskQueryModel(nn.Module):
    """
    Shared trunk -> two heads.

    Trunk: 20 -> 64 -> 32 (with dropout)
    Classification head: 32 -> 16 -> 1  (logit; sigmoid applied at inference)
    Regression head:     32 -> 16 -> 1  (predicts log1p of execution time in ms)
    """
    def __init__(self, n_features=20, dropout=DROPOUT):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head_clf = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.head_reg = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        z = self.trunk(x)
        return self.head_clf(z), self.head_reg(z)

model = MultiTaskQueryModel(n_features=len(FEATURE_COLS))

slow_count = int((y_clf_train == 1).sum())
fast_count = int((y_clf_train == 0).sum())
pos_weight = torch.tensor([fast_count / max(slow_count, 1)], dtype=torch.float32)
print(f"  pos_weight (fast/slow): {pos_weight.item():.2f}")

bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# mse = nn.MSELoss()
def quantile_loss(pred, target, tau=0.7):
    """
    Quantile (pinball) loss. tau > 0.5 means we penalize under-prediction
    more than over-prediction, biasing the model to predict on the high
    side of the true value. tau = 0.5 is equivalent to MAE / median
    regression.
    """
    diff = target - pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))

REG_TAU = 0.7  # quantile to target; >0.5 biases predictions upward

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ───────────────────────── Train ─────────────────────────

print(f"\nTraining for {EPOCHS} epochs (alpha={ALPHA})...\n")

history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = {"total": 0.0, "bce": 0.0, "qloss": 0.0}
    for xb, yb_clf, yb_reg in train_loader:
        optimizer.zero_grad()
        logit, pred_reg = model(xb)
        loss_clf = bce(logit, yb_clf)
        # loss_reg = mse(pred_reg, yb_reg)
        loss_reg = quantile_loss(pred_reg, yb_reg, tau=REG_TAU)
        loss = ALPHA * loss_clf + (1 - ALPHA) * loss_reg
        loss.backward()
        optimizer.step()
        running["total"] += loss.item()
        running["bce"] += loss_clf.item()
        running["qloss"] += loss_reg.item()

    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            logit, pred_reg = model(X_test_t)
            prob = torch.sigmoid(logit)
            preds = (prob > 0.5).float()
            acc = (preds == y_clf_test_t).float().mean().item()

            pred_log = pred_reg.cpu().numpy().ravel()
            true_log = y_reg_test_t.cpu().numpy().ravel()
            mae_log = mean_absolute_error(true_log, pred_log)

        n = len(train_loader)
        print(f"Epoch {epoch:3d} | "
              f"loss={running['total']/n:.3f} "
              f"(bce={running['bce']/n:.3f}, qloss={running['qloss']/n:.3f}) | "
              f"test_acc={acc*100:.1f}%  log_mae={mae_log:.3f}")
        history.append({
            "epoch": epoch,
            "loss": running["total"] / n,
            "bce": running["bce"] / n,
            "qloss": running["qloss"] / n,
            "test_acc": acc,
            "log_mae": mae_log,
        })

# ───────────────────────── Final Evaluation ─────────────────────────

print("\n── Final Evaluation ──\n")
model.eval()
with torch.no_grad():
    logit, pred_reg = model(X_test_t)
    prob = torch.sigmoid(logit).cpu().numpy().ravel()
    preds = (prob > 0.5).astype(np.float32)
    y_true = y_clf_test_t.cpu().numpy().ravel()

    pred_log = pred_reg.cpu().numpy().ravel()
    true_log = y_reg_test_t.cpu().numpy().ravel()

    pred_ms = np.expm1(pred_log)
    true_ms = np.expm1(true_log)

print("Classification:")
print(classification_report(y_true, preds, target_names=["fast", "slow"], digits=3))
print("Confusion matrix [rows=true, cols=pred]:")
print(confusion_matrix(y_true, preds))

print("\nRegression (predicting execution time in ms):")
print(f"  log-space MAE: {mean_absolute_error(true_log, pred_log):.3f}")
print(f"  ms-space MAE:  {mean_absolute_error(true_ms, pred_ms):.1f} ms")
print(f"  R²:            {r2_score(true_log, pred_log):.3f}")

# ───────────────────────── Save ─────────────────────────

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

meta = {
    "feature_cols": FEATURE_COLS,
    "n_features": len(FEATURE_COLS),
    "alpha": ALPHA,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "training_history": history,
    "n_train": len(X_train_t),
    "n_test": len(X_test_t),
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nModel saved to {MODEL_PATH}")
print(f"Scaler saved to {SCALER_PATH}")
print(f"Metadata saved to {META_PATH}")