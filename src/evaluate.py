from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve

from src.config import settings
from src.data import load_metadata, make_loader
from src.model import build_model


def resolve_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def specificity_from_confusion(tn, fp):
    return tn / (tn + fp) if (tn + fp) else float("nan")


def calculate_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_from_confusion(tn, fp)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def collect_predictions(model, loader, device):
    y_true, y_prob, image_ids = [], [], []
    model.eval()
    with torch.no_grad():
        for images, labels, ids in loader:
            logits = model(images.to(device)).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true.extend(labels.numpy().astype(int).tolist())
            y_prob.extend(probs.tolist())
            image_ids.extend(list(ids))
    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float), image_ids


def select_validation_threshold(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    valid = np.isfinite(thresholds) & (thresholds >= 0.0) & (thresholds <= 1.0)
    if not valid.any():
        return 0.5
    valid_indices = np.flatnonzero(valid)
    youden = tpr[valid_indices] - fpr[valid_indices]
    best_local = int(np.argmax(youden))
    return float(thresholds[valid_indices[best_local]])


def save_confusion_matrix(cm, output_path):
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    image = ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["Non-referable", "Referable"])
    ax.set_yticks([0, 1], labels=["Non-referable", "Referable"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default=str(settings.metadata_path))
    parser.add_argument("--checkpoint", default=str(settings.model_path))
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=settings.batch_size)
    parser.add_argument("--threshold", type=float, default=settings.decision_threshold)
    parser.add_argument("--output-dir", default=str(settings.output_dir))
    args = parser.parse_args()

    device = resolve_device()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    image_size = int(checkpoint.get("image_size", settings.image_size))

    model = build_model(weights="none")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    frame = load_metadata(args.metadata)

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "configured"
        validation_selection = None
    else:
        val_loader = make_loader(frame, "val", args.batch_size, image_size, num_workers=settings.num_workers)
        val_true, val_prob, _ = collect_predictions(model, val_loader, device)
        threshold = select_validation_threshold(val_true, val_prob)
        validation_selection = calculate_metrics(val_true, val_prob, threshold)
        threshold_source = "validation_youden_j"
        checkpoint["threshold"] = threshold
        checkpoint["threshold_source"] = threshold_source
        checkpoint["validation_threshold_metrics"] = validation_selection
        torch.save(checkpoint, checkpoint_path)
        print(f"Selected validation threshold: {threshold:.4f}")

    loader = make_loader(frame, args.split, args.batch_size, image_size, num_workers=settings.num_workers)
    split_frame = frame[frame["split"].astype(str).str.lower() == args.split.lower()].copy().set_index("image_id")
    y_true, y_prob, image_ids = collect_predictions(model, loader, device)
    chosen = calculate_metrics(y_true, y_prob, threshold)
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "split": args.split,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "n": int(len(y_true)),
        "accuracy": chosen["accuracy"],
        "precision": chosen["precision"],
        "sensitivity_recall": chosen["sensitivity"],
        "specificity": chosen["specificity"],
        "f1": chosen["f1"],
        "roc_auc": roc_auc,
        "tn": chosen["tn"], "fp": chosen["fp"], "fn": chosen["fn"], "tp": chosen["tp"],
        "model": {
            "architecture": checkpoint.get("architecture", "resnet18"),
            "image_size": image_size,
            "best_epoch": checkpoint.get("best_epoch"),
            "best_val_auc": checkpoint.get("best_val_auc"),
            "positive_class_weight": checkpoint.get("positive_class_weight"),
            "weights_initialization": checkpoint.get("weights_initialization"),
        },
    }
    if validation_selection is not None:
        metrics["validation_threshold_metrics"] = validation_selection

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    predictions = pd.DataFrame({"image_id": image_ids, "target": y_true, "probability_referable": y_prob, "prediction": (y_prob >= threshold).astype(int)})
    if "dr_grade" in split_frame.columns:
        predictions["dr_grade"] = predictions["image_id"].map(split_frame["dr_grade"])
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    save_confusion_matrix(cm, output_dir / "confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=170)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall / Sensitivity")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=170)
    plt.close(fig)

    if settings.history_path.exists():
        try:
            history = json.loads(settings.history_path.read_text())
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
            fig.savefig(settings.history_path.with_suffix(".png"), dpi=170)
            plt.close(fig)
        except Exception as exc:
            print(f"Warning: could not create training-history plot: {exc}")

    print(json.dumps(metrics, indent=2))
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
