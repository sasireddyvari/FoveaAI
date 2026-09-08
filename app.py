from __future__ import annotations

import base64
import hashlib
import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

from src.config import settings
from src.dashboard import dashboard_payload
from src.predict import load_predictor, predict_pil


app = Flask(__name__)
app.secret_key = settings.secret_key
app.config["MAX_CONTENT_LENGTH"] = settings.max_upload_mb * 1024 * 1024
prediction_lock = threading.Lock()
predictor = None
startup_error = None


def initialize_predictor():
    global predictor, startup_error
    if not settings.model_path.exists():
        startup_error = f"Checkpoint not found: {settings.model_path}"
        predictor = None
        return
    try:
        predictor = load_predictor(settings.model_path, settings.decision_threshold)
        startup_error = None
    except Exception as exc:
        predictor = None
        startup_error = str(exc)


def initialize_database():
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_ts TEXT NOT NULL,
                source_token TEXT NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                threshold REAL NOT NULL,
                confidence REAL NOT NULL,
                technical_status TEXT NOT NULL,
                latency_ms REAL NOT NULL
            )
        """)
        connection.commit()


def log_prediction(file_bytes, result):
    if not settings.store_prediction_logs:
        return
    source_token = hashlib.sha256(file_bytes).hexdigest()[:12]
    with sqlite3.connect(settings.database_path) as connection:
        connection.execute(
            "INSERT INTO prediction_log(event_ts, source_token, prediction, probability, threshold, confidence, technical_status, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(), source_token, result["label"], result["probability"], result["threshold"],
                result["confidence_percent"] / 100.0, result["quality"]["technical_status"], result["latency_ms"],
            ),
        )
        connection.commit()


def recent_predictions():
    if not settings.database_path.exists():
        return []
    with sqlite3.connect(settings.database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT event_ts, source_token, prediction, probability, threshold, confidence, technical_status, latency_ms FROM prediction_log ORDER BY id DESC LIMIT ?",
            (settings.recent_predictions_limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def image_to_data_uri(image, fmt="JPEG"):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt, quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def execute_prediction(file_bytes):
    if predictor is None:
        raise RuntimeError(startup_error or "Model is not loaded.")
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    model, transform, threshold, device, checkpoint = predictor
    with prediction_lock:
        result = predict_pil(image, model, transform, threshold, device, with_gradcam=settings.enable_gradcam)
    result["preview_uri"] = image_to_data_uri(image)
    overlay = result.pop("gradcam", None)
    result["gradcam_uri"] = image_to_data_uri(overlay) if overlay is not None else None
    result["device"] = str(device)
    result["architecture"] = checkpoint.get("architecture", "resnet18")
    return result


initialize_database()
initialize_predictor()


@app.route("/", methods=["GET", "POST"])
def prediction_view():
    result, error = None, startup_error
    if request.method == "POST":
        upload = request.files.get("image")
        if upload is None or upload.filename == "":
            error = "Choose a retinal image before running the model."
        else:
            try:
                file_bytes = upload.read()
                result = execute_prediction(file_bytes)
                log_prediction(file_bytes, result)
                error = None
            except UnidentifiedImageError:
                error = "The uploaded file is not a supported image."
            except Exception as exc:
                error = f"Prediction failed: {exc}"
    return render_template("prediction.html", active="prediction", result=result, error=error, model_ready=predictor is not None, current_threshold=(predictor[2] if predictor is not None else settings.decision_threshold or 0.5), recent=recent_predictions(), settings=settings)


@app.route("/metrics")
def metrics_view():
    return render_template("metrics.html", active="metrics", dashboard=dashboard_payload(), settings=settings)


@app.route("/artifact/<name>")
def artifact(name):
    allowed = {
        "roc": settings.output_dir / "roc_curve.png",
        "pr": settings.output_dir / "precision_recall_curve.png",
        "confusion": settings.output_dir / "confusion_matrix.png",
        "training": settings.history_path.with_suffix(".png"),
    }
    path = allowed.get(name)
    if path is None or not path.exists():
        return jsonify({"error": "Artifact not found"}), 404
    return send_file(path)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify({"error": "image file is required"}), 400
    try:
        file_bytes = upload.read()
        result = execute_prediction(file_bytes)
        log_prediction(file_bytes, result)
        result.pop("preview_uri", None)
        result.pop("gradcam_uri", None)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if predictor is not None else "degraded",
        "model_ready": predictor is not None,
        "model_path": str(settings.model_path),
        "metrics_ready": (settings.output_dir / "metrics.json").exists(),
        "gradcam_enabled": settings.enable_gradcam,
    })


if __name__ == "__main__":
    app.run(host=settings.host, port=settings.port, debug=False)
