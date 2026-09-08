from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from src.config import settings


def load_json(path: Path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except Exception:
        return None


def dataset_summary():
    if not settings.metadata_path.exists():
        return None
    frame = pd.read_csv(settings.metadata_path)
    splits = frame.groupby(["split", "target"]).size().reset_index(name="count").to_dict("records")
    grades = frame.groupby(["split", "dr_grade"]).size().reset_index(name="count").to_dict("records") if "dr_grade" in frame.columns else []
    return {
        "total": int(len(frame)),
        "train": int((frame["split"] == "train").sum()),
        "val": int((frame["split"] == "val").sum()),
        "test": int((frame["split"] == "test").sum()),
        "splits": splits,
        "grades": grades,
    }


def model_metadata():
    if not settings.model_path.exists():
        return None
    try:
        checkpoint = torch.load(settings.model_path, map_location="cpu")
        history = load_json(settings.history_path) or []
        best_history = max(history, key=lambda row: row.get("val_auc", -1), default={})
        return {
            "architecture": checkpoint.get("architecture", "resnet18"),
            "image_size": checkpoint.get("image_size", 224),
            "threshold": settings.decision_threshold if settings.decision_threshold is not None else checkpoint.get("threshold", 0.5),
            "threshold_source": checkpoint.get("threshold_source", "configured" if settings.decision_threshold is not None else "checkpoint"),
            "best_epoch": checkpoint.get("best_epoch") or best_history.get("epoch"),
            "best_val_auc": checkpoint.get("best_val_auc") or best_history.get("val_auc"),
            "positive_class_weight": checkpoint.get("positive_class_weight"),
            "weights_initialization": checkpoint.get("weights_initialization"),
        }
    except Exception:
        return None


def dashboard_payload():
    return {
        "metrics": load_json(settings.output_dir / "metrics.json"),
        "history": load_json(settings.history_path),
        "dataset": dataset_summary(),
        "model": model_metadata(),
        "artifacts": {
            "roc": (settings.output_dir / "roc_curve.png").exists(),
            "pr": (settings.output_dir / "precision_recall_curve.png").exists(),
            "confusion": (settings.output_dir / "confusion_matrix.png").exists(),
            "training": settings.history_path.with_suffix(".png").exists(),
        },
    }
