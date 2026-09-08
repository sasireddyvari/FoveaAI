from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _path(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default)).expanduser()
    return value if value.is_absolute() else PROJECT_ROOT / value


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _optional_float(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    app_name: str = os.getenv("APP_NAME", "FoveaAI")
    app_env: str = os.getenv("APP_ENV", "local")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "5050"))
    secret_key: str = os.getenv("SECRET_KEY", "foveaai-local-research")
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "12"))

    idrid_root: Path = _path("IDRID_ROOT", "data/raw/IDRiD")
    train_images: Path = _path("TRAIN_IMAGES", "data/raw/IDRiD/B. Disease Grading/1. Original Images/a. Training Set")
    train_labels: Path = _path("TRAIN_LABELS", "data/raw/IDRiD/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv")
    test_images: Path = _path("TEST_IMAGES", "data/raw/IDRiD/B. Disease Grading/1. Original Images/b. Testing Set")
    test_labels: Path = _path("TEST_LABELS", "data/raw/IDRiD/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv")

    metadata_path: Path = _path("METADATA_PATH", "data/processed/metadata.csv")
    model_path: Path = _path("MODEL_PATH", "models/foveaai_resnet18.pt")
    history_path: Path = _path("HISTORY_PATH", "models/foveaai_resnet18.history.json")
    output_dir: Path = _path("OUTPUT_DIR", "outputs/test")
    database_path: Path = _path("DATABASE_PATH", "instance/foveaai.db")

    epochs: int = int(os.getenv("EPOCHS", "12"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "16"))
    learning_rate: float = float(os.getenv("LEARNING_RATE", "0.0001"))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", "0.0001"))
    image_size: int = int(os.getenv("IMAGE_SIZE", "224"))
    num_workers: int = int(os.getenv("NUM_WORKERS", "0"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))
    model_weights: str = os.getenv("MODEL_WEIGHTS", "imagenet")

    decision_threshold: float | None = _optional_float("DECISION_THRESHOLD")
    enable_gradcam: bool = _bool("ENABLE_GRADCAM", True)
    store_prediction_logs: bool = _bool("STORE_PREDICTION_LOGS", True)
    recent_predictions_limit: int = int(os.getenv("RECENT_PREDICTIONS_LIMIT", "20"))


settings = Settings()
