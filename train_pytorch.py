import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", str(text).lower())


def build_vocab(texts: pd.Series, max_vocab: int = 20000, min_freq: int = 2) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common(max_vocab - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


class TicketDataset(Dataset):
    def __init__(self, texts: pd.Series, labels: list[int], vocab: dict[str, int]):
        self.encoded = [
            [vocab.get(tok, 1) for tok in tokenize(t)] or [1] for t in texts
        ]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return torch.tensor(self.encoded[idx], dtype=torch.long), self.labels[idx]


def collate_bag(batch):
    offsets = [0]
    tokens = []
    labels = []
    for seq, label in batch:
        tokens.append(seq)
        offsets.append(offsets[-1] + len(seq))
        labels.append(label)
    flat_tokens = torch.cat(tokens)
    offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return flat_tokens, offsets_tensor, labels_tensor


class SmallTicketClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, text: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(text, offsets))
        return self.fc(embedded)


def train_and_compare() -> None:
    device = torch.device("cpu")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    labels = sorted(train_df["category"].unique())
    label_to_idx = {c: i for i, c in enumerate(labels)}

    # Build vocab strictly on training tickets (no data leakage)
    vocab = build_vocab(train_df["ticket_text"])

    train_ds = TicketDataset(
        train_df["ticket_text"],
        [label_to_idx[c] for c in train_df["category"]],
        vocab,
    )
    test_ds = TicketDataset(
        test_df["ticket_text"],
        [label_to_idx[c] for c in test_df["category"]],
        vocab,
    )

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, collate_fn=collate_bag)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, collate_fn=collate_bag)

    # Inverse-frequency class weights to handle class imbalance fairly
    counts = train_df["category"].value_counts()
    weights = torch.tensor(
        [len(train_df) / (len(labels) * counts[c]) for c in labels],
        dtype=torch.float32,
    )

    model = SmallTicketClassifier(len(vocab), embed_dim=64, num_classes=len(labels)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    # Train for 8 epochs (takes ~10-15 seconds on CPU)
    epochs = 8
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for flat_tokens, offsets, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(flat_tokens, offsets)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{epochs} - Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate on unseen test set & measure inference latency
    model.eval()
    all_preds = []
    all_confs = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for flat_tokens, offsets, _ in test_loader:
            logits = model(flat_tokens, offsets)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)
            all_preds.extend(preds.tolist())
            all_confs.extend(confs.tolist())
    total_ms = (time.perf_counter() - start_time) * 1000
    avg_latency_ms = total_ms / len(test_df)

    y_true = [label_to_idx[c] for c in test_df["category"]]
    acc = accuracy_score(y_true, all_preds)
    macro_f1 = f1_score(y_true, all_preds, average="macro")

    print("\n=== MODEL 2: SMALL PYTORCH CLASSIFIER ===")
    print(f"Accuracy          : {acc:.4f}")
    print(f"Macro F1          : {macro_f1:.4f}")
    print(f"Avg Latency/Ticket: {avg_latency_ms:.4f} ms")
    print(f"Mean Max Conf     : {np.mean(all_confs):.4f}\n")

    print("=== PER-CLASS REPORT ===")
    print(classification_report(y_true, all_preds, target_names=labels, digits=4))

    # Save PyTorch artifacts
    pt_path = MODELS_DIR / "pytorch_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab),
            "embed_dim": 64,
            "classes": labels,
        },
        pt_path,
    )
    with open(MODELS_DIR / "pytorch_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    pt_metrics = {
        "model_name": "Small PyTorch EmbeddingBag Classifier",
        "model_version": "v1.0-pytorch",
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "avg_latency_ms": round(float(avg_latency_ms), 4),
        "classes": labels,
    }
    with open(MODELS_DIR / "pytorch_metrics.json", "w", encoding="utf-8") as f:
        json.dump(pt_metrics, f, indent=2)

    # Side-by-side comparison with Model 1
    logreg_metrics_path = MODELS_DIR / "logreg_metrics.json"
    if logreg_metrics_path.exists():
        with open(logreg_metrics_path, "r", encoding="utf-8") as f:
            lr_metrics = json.load(f)

        lr_size_kb = (
            (MODELS_DIR / "tfidf_vectorizer.joblib").stat().st_size
            + (MODELS_DIR / "logreg_model.joblib").stat().st_size
        ) / 1024
        pt_size_kb = (
            pt_path.stat().st_size + (MODELS_DIR / "pytorch_vocab.json").stat().st_size
        ) / 1024

        print("=== EVIDENCE-BASED MODEL COMPARISON ===")
        print(f"{'Metric':<20} | {'TF-IDF + LogReg':<18} | {'PyTorch Classifier':<18}")
        print("-" * 62)
        print(f"{'Accuracy':<20} | {lr_metrics['accuracy']:<18.4f} | {pt_metrics['accuracy']:<18.4f}")
        print(f"{'Macro F1':<20} | {lr_metrics['macro_f1']:<18.4f} | {pt_metrics['macro_f1']:<18.4f}")
        print(f"{'Latency (ms/ticket)':<20} | {lr_metrics['avg_latency_ms']:<18.4f} | {pt_metrics['avg_latency_ms']:<18.4f}")
        print(f"{'Artifact Size (KB)':<20} | {lr_size_kb:<18.1f} | {pt_size_kb:<18.1f}")


if __name__ == "__main__":
    train_and_compare()