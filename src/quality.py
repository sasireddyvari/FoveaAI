from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def technical_quality_from_pil(image: Image.Image):
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    field_coverage = float((gray > 15).mean() * 100)

    flags = []
    if min(width, height) < 224:
        flags.append("Low source resolution")
    if sharpness < 20:
        flags.append("Very low sharpness")
    elif sharpness < 60:
        flags.append("Low sharpness")
    if brightness < 35:
        flags.append("Image appears dark")
    elif brightness > 220:
        flags.append("Image appears bright")
    if contrast < 20:
        flags.append("Low contrast")
    if field_coverage < 35:
        flags.append("Limited visible retinal field")

    return {
        "width": int(width),
        "height": int(height),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "field_coverage": round(field_coverage, 2),
        "technical_status": "Review" if flags else "Pass",
        "flags": flags,
    }
