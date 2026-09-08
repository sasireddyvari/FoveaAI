import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.config import settings


checks = [
    ("Training images", settings.train_images),
    ("Training labels", settings.train_labels),
    ("Testing images", settings.test_images),
    ("Testing labels", settings.test_labels),
    ("Metadata", settings.metadata_path),
    ("Model checkpoint", settings.model_path),
    ("Evaluation metrics", settings.output_dir / "metrics.json"),
]

print("FoveaAI setup check")
print("=" * 64)
for label, path in checks:
    print(f"{label:22s} {'OK' if Path(path).exists() else 'MISSING':8s} {path}")

if torch.backends.mps.is_available():
    device = "Apple MPS"
elif torch.cuda.is_available():
    device = "CUDA"
else:
    device = "CPU"
print(f"{'PyTorch device':22s} {device}")
print(f"{'Web app':22s} http://127.0.0.1:{settings.port}")
