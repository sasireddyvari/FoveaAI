from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class RetinalDataset(Dataset):
    def __init__(self, frame, transform=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(Path(row["image_path"])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(row["target"]), dtype=torch.float32)
        return image, label, str(row["image_id"])


def load_metadata(path):
    frame = pd.read_csv(path)
    required = {"image_id", "image_path", "target", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Metadata is missing required columns: {sorted(missing)}")
    frame["target"] = frame["target"].astype(int)
    return frame


def make_loader(frame, split, batch_size=16, image_size=224, num_workers=0):
    subset = frame[frame["split"].astype(str).str.lower() == split.lower()].copy()
    if subset.empty:
        raise ValueError(f"No rows found for split='{split}'")
    transform = get_train_transform(image_size) if split.lower() == "train" else get_eval_transform(image_size)
    dataset = RetinalDataset(subset, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split.lower() == "train", num_workers=num_workers, pin_memory=torch.cuda.is_available())


def calculate_pos_weight(frame):
    train = frame[frame["split"].astype(str).str.lower() == "train"]
    negatives = int((train["target"] == 0).sum())
    positives = int((train["target"] == 1).sum())
    if positives == 0:
        raise ValueError("Training split contains no positive samples.")
    return negatives / positives
