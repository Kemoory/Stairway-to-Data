import torch
import torch.nn as nn
from torchvision.models import resnet18, ViT_B_16_Weights, vit_b_16
import time

class SimpleCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config['channels']
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.regressor = nn.Sequential(
            nn.Dropout(config['dropout']),
            nn.Linear(channels[2], 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.regressor(x.view(x.size(0), -1))

class ResNet18(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = resnet18(pretrained=config['pretrained'])
        if config['freeze_backbone']:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if config['pretrained'] else None)
        self.model.heads = nn.Linear(self.model.heads.head.in_features, 1)

    def forward(self, x):
        return self.model(x)

def get_model(model_type, config):
    if model_type == 'resnet18':
        return ResNet18(config)
    elif model_type == 'simple_cnn':
        return SimpleCNN(config)
    elif model_type == 'vit':
        return VisionTransformer(config)
    raise ValueError(f"Modèle non supporté: {model_type}")