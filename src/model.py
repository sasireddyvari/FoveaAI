import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(weights="imagenet"):
    selected_weights = ResNet18_Weights.DEFAULT if weights == "imagenet" else None
    model = resnet18(weights=selected_weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
