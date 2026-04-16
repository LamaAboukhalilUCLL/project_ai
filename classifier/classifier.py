import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle

df = pd.read_csv("training_data.csv")

feature_cols = [
    "has_select_star",
    "has_like_wildcard",
    "join_count",
    "has_subquery",
    "has_group_by",
    "has_order_by_no_limit",
    "has_or",
    "has_function_in_where"
]

X = df[feature_cols].values.astype(np.float32)
y = (df["label"] == "slow").astype(np.float32).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=16, shuffle=True)

class QueryClassifier(nn.Module):
    def __init__(self):
        super(QueryClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = QueryClassifier()

slow_count = (y_train == 1).sum()
fast_count = (y_train == 0).sum()
pos_weight = torch.tensor([fast_count / slow_count], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training...\n")
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds = (model(X_test_t) > 0.5).float()
            correct = (preds == y_test_t).float().mean()
            print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Test Accuracy: {correct.item()*100:.1f}%")

print("\n── Final Evaluation ──")
model.eval()
with torch.no_grad():
    preds = (model(X_test_t) > 0.5).float().numpy()
    y_true = y_test_t.numpy()

print(classification_report(y_true, preds, target_names=["fast", "slow"]))
print("Confusion matrix:")
print(confusion_matrix(y_true, preds))

torch.save(model.state_dict(), "classifier/query_classifier.pth")
with open("classifier/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel saved to query_classifier.pth")
print("Scaler saved to scaler.pkl")