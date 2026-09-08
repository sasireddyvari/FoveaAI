from __future__ import annotations

import argparse
import json
import time

import torch
from PIL import Image

from src.data import get_eval_transform
from src.gradcam import gradcam_overlay
from src.model import build_model
from src.quality import technical_quality_from_pil


def load_predictor(checkpoint_path, threshold_override=None, device=None):
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(weights="none")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    image_size = int(checkpoint.get("image_size", 224))
    checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    threshold = float(threshold_override) if threshold_override is not None else checkpoint_threshold
    transform = get_eval_transform(image_size)
    return model, transform, threshold, device, checkpoint


def predict_pil(image, model, transform, threshold, device, with_gradcam=True):
    started = time.perf_counter()
    image = image.convert("RGB")
    quality = technical_quality_from_pil(image)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor).squeeze()
        probability = float(torch.sigmoid(logit).item())

    predicted_class = int(probability >= threshold)
    label = "Referable DR" if predicted_class == 1 else "Non-referable DR"

    # Keep this legacy score for API/database compatibility only. It is not shown as
    # "confidence" in the UI because the model probability has not been calibrated.
    class_score = probability if predicted_class == 1 else 1.0 - probability

    signed_margin = probability - threshold
    decision_margin = abs(signed_margin)
    margin_position = "above" if signed_margin >= 0 else "below"
    decision_margin_text = f"{decision_margin * 100:.1f} pp {margin_position} threshold"

    overlay = gradcam_overlay(model, tensor, image, device) if with_gradcam else None
    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "label": label,
        "predicted_class": predicted_class,
        "probability": probability,
        "probability_percent": round(probability * 100, 2),
        "confidence_percent": round(class_score * 100, 2),
        "threshold": threshold,
        "threshold_percent": round(threshold * 100, 2),
        "margin": round(decision_margin, 4),
        "decision_margin_percent": round(decision_margin * 100, 2),
        "decision_margin_text": decision_margin_text,
        "quality": quality,
        "latency_ms": round(latency_ms, 2),
        "gradcam": overlay,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--no-gradcam", action="store_true")
    args = parser.parse_args()

    model, transform, threshold, device, _ = load_predictor(args.checkpoint, args.threshold)
    image = Image.open(args.image).convert("RGB")
    result = predict_pil(image, model, transform, threshold, device, with_gradcam=not args.no_gradcam)
    result.pop("gradcam", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
