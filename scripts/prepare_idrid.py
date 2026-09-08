from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import settings


IMAGE_COLUMN_CANDIDATES = ["Image name", "image_name", "image", "Image"]
GRADE_COLUMN_CANDIDATES = ["Retinopathy grade", "retinopathy_grade", "grade", "DR grade"]


def find_column(frame, candidates):
    mapping = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in mapping:
            return mapping[candidate.lower()]
    raise ValueError(f"Could not find any of {candidates}. CSV columns are: {list(frame.columns)}")


def build_frame(images_dir, labels_csv, official_split):
    labels = pd.read_csv(labels_csv)
    image_column = find_column(labels, IMAGE_COLUMN_CANDIDATES)
    grade_column = find_column(labels, GRADE_COLUMN_CANDIDATES)
    rows = []
    images_dir = Path(images_dir)

    for _, row in labels.iterrows():
        image_id = str(row[image_column]).strip()
        grade = int(row[grade_column])
        candidates = [images_dir / image_id, images_dir / f"{image_id}.jpg", images_dir / f"{image_id}.jpeg", images_dir / f"{image_id}.png"]
        image_path = next((path for path in candidates if path.exists()), None)
        if image_path is None:
            raise FileNotFoundError(f"Image not found for '{image_id}' under {images_dir}")
        rows.append({"image_id": Path(image_id).stem, "image_path": str(image_path.resolve()), "dr_grade": grade, "target": int(grade >= 2), "official_split": official_split})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-images", default=str(settings.train_images))
    parser.add_argument("--train-labels", default=str(settings.train_labels))
    parser.add_argument("--test-images", default=str(settings.test_images))
    parser.add_argument("--test-labels", default=str(settings.test_labels))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    parser.add_argument("--output", default=str(settings.metadata_path))
    args = parser.parse_args()

    train_all = build_frame(args.train_images, args.train_labels, "train")
    test = build_frame(args.test_images, args.test_labels, "test")
    train, val = train_test_split(train_all, test_size=args.val_size, random_state=args.seed, stratify=train_all["target"])
    train, val, test = train.copy(), val.copy(), test.copy()
    train["split"], val["split"], test["split"] = "train", "val", "test"
    combined = pd.concat([train, val, test], ignore_index=True)

    duplicate_mask = combined.duplicated(subset=["split", "image_id"], keep=False)
    if duplicate_mask.any():
        duplicates = combined.loc[duplicate_mask, ["split", "image_id"]]
        raise ValueError(f"Duplicate image_id values detected within the same split:\n{duplicates.to_string(index=False)}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"Saved: {output}")
    print(combined.groupby(["split", "target"]).size().to_string())
    print("\nDR grade distribution:")
    print(combined.groupby(["split", "dr_grade"]).size().to_string())


if __name__ == "__main__":
    main()
