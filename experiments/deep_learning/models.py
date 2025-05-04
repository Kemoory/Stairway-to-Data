import torch
import torch.nn as nn
from torchvision.models import resnet18, ViT_B_16_Weights, vit_b_16
import time
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

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
        weights = ResNet18_Weights.IMAGENET1K_V1 if config['pretrained'] else None
        self.model = resnet18(weights=weights)
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

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class StairFeatureExtractor(nn.Module):
    def __init__(self):
        super(StairFeatureExtractor, self).__init__()
        # Encoder path (downsampling)
        self.enc1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )
        
    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Bridge
        b = self.bridge(p4)
        
        # Return multi-scale features
        return b, e4, e3, e2, e1


class StairNetDepth(nn.Module):
    def __init__(self, config):
        super(StairNetDepth, self).__init__()
        
        # Feature extractor backbone
        self.feature_extractor = StairFeatureExtractor()
        
        # Global average pooling and regression
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Attention modules for different scales
        self.attention1 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, 1),
            nn.Sigmoid()
        )
        
        self.attention2 = nn.Sequential(
            nn.Conv2d(512, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 512, 1),
            nn.Sigmoid()
        )
        
        # Final regression layers
        self.fc1 = nn.Linear(1408, 256)  # 512 + 256 + 128 + 64 = 960
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Extract multi-scale features
        b, e4, e3, e2, e1 = self.feature_extractor(x)
        
        # Apply attention to bridge features
        att1 = self.attention1(b)
        b_att = b * att1
        
        # Apply attention to e4 features
        att2 = self.attention2(e4)
        e4_att = e4 * att2
        
        # Global average pooling on attended features
        b_feat = self.global_pool(b_att).view(x.size(0), -1)  # 512
        e4_feat = self.global_pool(e4_att).view(x.size(0), -1)  # 256
        e3_feat = self.global_pool(e3).view(x.size(0), -1)  # 256
        e2_feat = self.global_pool(e2).view(x.size(0), -1)  # 128
        
        # S'assurer que toutes les dimensions sont correctes avant la concaténation
        # Prendre les 128 premiers canaux de e3 et 64 premiers de e2 si nécessaire
        e3_feat_slice = e3_feat[:, :128] if e3_feat.size(1) > 128 else e3_feat
        e2_feat_slice = e2_feat[:, :64] if e2_feat.size(1) > 64 else e2_feat
        
        # Redimensionner les tenseurs si nécessaire
        if e3_feat_slice.size(1) < 128:
            padding = torch.zeros(x.size(0), 128 - e3_feat_slice.size(1), device=x.device)
            e3_feat_slice = torch.cat([e3_feat_slice, padding], dim=1)
            
        if e2_feat_slice.size(1) < 64:
            padding = torch.zeros(x.size(0), 64 - e2_feat_slice.size(1), device=x.device)
            e2_feat_slice = torch.cat([e2_feat_slice, padding], dim=1)
        
        # Concatenate features for regression
        combined = torch.cat([b_feat, e4_feat, e3_feat, e2_feat], dim=1)  # Sans troncature
        
        # Regression path
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def get_model(model_type, config):
    if model_type == 'resnet18':
        return ResNet18(config)
    elif model_type == 'simple_cnn':
        return SimpleCNN(config)
    elif model_type == 'vit':
        return VisionTransformer(config)
    elif model_type == 'stairnet_depth':
        return StairNetDepth(config)
    raise ValueError(f"Modèle non supporté: {model_type}")