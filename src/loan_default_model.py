"""Loan default prediction with a PyTorch DNN and logistic regression baseline."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path("data/early_2012_2013_loan_sample_with_outcome.csv")
TARGET_COL = "loan_is_bad"
RANDOM_SEED = 42


class LogisticRegressionBaseline(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LoanDefaultDNN(nn.Module):
    def __init__(self, input_dim: int, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_and_prepare_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    leakage_cols = [
        "loan_status",
        "out_prncp",
        "out_prncp_inv",
        "total_pymnt",
        "total_pymnt_inv",
        "total_rec_prncp",
        "total_rec_int",
        "total_rec_late_fee",
        "recoveries",
        "collection_recovery_fee",
        "last_pymnt_d",
        "last_pymnt_amnt",
        "next_pymnt_d",
        "last_credit_pull_d",
    ]
    noisy_cols = ["id", "member_id", "zip_code", "emp_title", "desc", "title"]
    redundant_cols = ["grade", "funded_amnt", "funded_amnt_inv"]
    high_missing_cols = [c for c in df.columns if c != TARGET_COL and df[c].isna().mean() > 0.60]

    drop_cols = [c for c in leakage_cols + noisy_cols + redundant_cols + high_missing_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    for col in ["term", "int_rate", "revol_util", "emp_length"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("months", "", regex=False)
                .str.replace("month", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.replace("10+ years", "10", regex=False)
                .str.replace("< 1 year", "0", regex=False)
                .str.extract(r"([0-9.]+)")[0]
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if {"loan_amnt", "annual_inc"}.issubset(df.columns):
        df["loan_to_income"] = df["loan_amnt"] / df["annual_inc"].replace(0, np.nan)

    if {"installment", "annual_inc"}.issubset(df.columns):
        df["installment_to_income"] = df["installment"] / df["annual_inc"].replace(0, np.nan)

    if {"dti", "revol_util"}.issubset(df.columns):
        df["dti_x_revol_util"] = df["dti"] * df["revol_util"]

    df = df.replace([np.inf, -np.inf], np.nan)

    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])
    return x, y


def preprocess(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = x_train.select_dtypes(exclude=[np.number]).columns.tolist()

    train_medians = x_train[numeric_cols].median()
    x_train[numeric_cols] = x_train[numeric_cols].fillna(train_medians)
    x_val[numeric_cols] = x_val[numeric_cols].fillna(train_medians)
    x_test[numeric_cols] = x_test[numeric_cols].fillna(train_medians)

    x_train[categorical_cols] = x_train[categorical_cols].fillna("Missing")
    x_val[categorical_cols] = x_val[categorical_cols].fillna("Missing")
    x_test[categorical_cols] = x_test[categorical_cols].fillna("Missing")

    x_train_enc = pd.get_dummies(x_train, columns=categorical_cols)
    x_val_enc = pd.get_dummies(x_val, columns=categorical_cols).reindex(columns=x_train_enc.columns, fill_value=0)
    x_test_enc = pd.get_dummies(x_test, columns=categorical_cols).reindex(columns=x_train_enc.columns, fill_value=0)

    scaler = StandardScaler()
    return (
        scaler.fit_transform(x_train_enc),
        scaler.transform(x_val_enc),
        scaler.transform(x_test_enc),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    max_epochs: int = 50,
    patience: int = 5,
) -> nn.Module:
    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, criterion)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            losses.append(criterion(model(xb), yb).item() * len(xb))
    return sum(losses) / len(loader.dataset)


def predict_probabilities(model: nn.Module, x_tensor: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x_tensor)).numpy().ravel()


def metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, probs),
        "pr_auc": average_precision_score(y_true, probs),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
    }


def main() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    x, y = load_and_prepare_data(DATA_PATH)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )

    x_train_scaled, x_val_scaled, x_test_scaled = preprocess(x_train, x_val, x_test)

    x_train_t = torch.tensor(x_train_scaled, dtype=torch.float32)
    x_val_t = torch.tensor(x_val_scaled, dtype=torch.float32)
    x_test_t = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(x_train_t, y_train_t), batch_size=256, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val_t, y_val_t), batch_size=256, shuffle=False)

    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    models = {
        "DNN": LoanDefaultDNN(input_dim=x_train_t.shape[1]),
        "Logistic Regression Baseline": LogisticRegressionBaseline(input_dim=x_train_t.shape[1]),
    }

    for name, model in models.items():
        optimizer = optim.AdamW(model.parameters(), lr=0.001) if name == "DNN" else optim.SGD(model.parameters(), lr=0.01)
        trained_model = train_model(model, train_loader, val_loader, criterion, optimizer)
        test_probs = predict_probabilities(trained_model, x_test_t)
        print(name, metrics(y_test.values, test_probs))


if __name__ == "__main__":
    main()
