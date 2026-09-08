from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from src.config import settings
from src.data import calculate_pos_weight, load_metadata, make_loader
from src.model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    labels_all, probs_all = [], []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad()

            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            total_loss += loss.item() * images.size(0)
            labels_all.extend(labels.detach().cpu().numpy().tolist())
            probs_all.extend(probs.detach().cpu().numpy().tolist())

    mean_loss = total_loss / len(loader.dataset)
    auc = roc_auc_score(labels_all, probs_all) if len(set(labels_all)) > 1 else float("nan")
    return mean_loss, auc


def save_training_plot(history, path):
    epochs = [row["epoch"] for row in history]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, [row["train_auc"] for row in history], marker="o", label="Train AUC")
    ax.plot(epochs, [row["val_auc"] for row in history], marker="o", label="Validation AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Training and Validation ROC-AUC")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(settings.metadata_path))
    parser.add_argument("--epochs", type=int, default=settings.epochs)
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--lr", type=float, default=settings.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=settings.weight_decay)
    parser.add_argument("--weights", choices=["imagenet", "none"], default=settings.model_weights)
    parser.add_argument("--image-size", type=int, default=settings.image_size)
    parser.add_argument("--num-workers", type=int, default=settings.num_workers)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--output", default=str(settings.model_path))
    parser.add_argument("--history", default=str(settings.history_path))
    args = parser.parse_args()

    set_seed(args.seed)
    frame = load_metadata(args.metadata)
    device = resolve_device()
    train_loader = make_loader(frame, "train", args.batch_size, args.image_size, args.num_workers)
    val_loader = make_loader(frame, "val", args.batch_size, args.image_size, args.num_workers)
    pos_weight_value = calculate_pos_weight(frame)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    model = build_model(args.weights).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output = Path(args.output)
    history_path = Path(args.history)
    output.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_auc = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print(f"Device: {device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Positive class weight: {pos_weight_value:.4f}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_loss, val_auc = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        record = {"epoch": epoch, "train_loss": train_loss, "train_auc": train_auc, "val_loss": val_loss, "val_auc": val_auc}
        history.append(record)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} train_auc={train_auc:.4f} | val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        score = val_auc if np.isfinite(val_auc) else -1.0
        if score > best_auc:
            best_auc = score
            best_epoch = epoch
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "architecture": "resnet18",
                "image_size": args.image_size,
                "class_names": ["Non-referable DR", "Referable DR"],
                "threshold": 0.5,
                "weights_initialization": args.weights,
                "positive_class_weight": pos_weight_value,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_val_auc": best_auc,
            }
            torch.save(checkpoint, output)
            print(f"Saved best checkpoint -> {output}")
        else:
            epochs_without_improvement += 1

        history_path.write_text(json.dumps(history, indent=2))
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}; best validation AUC was {best_auc:.4f} at epoch {best_epoch}.")
            break

    plot_path = history_path.with_suffix(".png")
    save_training_plot(history, plot_path)
    print(f"Training history -> {history_path}")
    print(f"Training curve -> {plot_path}")


if __name__ == "__main__":
    main()
