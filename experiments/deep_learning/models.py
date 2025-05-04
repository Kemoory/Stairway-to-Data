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

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Block de convolution amélioré avec connexion résiduelle optionnelle"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_residual=False):
        super(ConvBlock, self).__init__()
        self.use_residual = use_residual and (in_channels == out_channels)
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        # On désactive inplace=True pour éviter les erreurs durant le backward pass
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        identity = x
        
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        if self.use_residual:
            out = out + identity  # Utilisation de out = out + identity au lieu de out += identity
            
        return out

class AttentionModule(nn.Module):
    """Module d'attention amélioré avec compression et expansion"""
    def __init__(self, in_channels, reduction_ratio=4):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = max(in_channels // reduction_ratio, 8)
        
        # Utiliser deux MLP séparés pour éviter les problèmes avec les graphes de calcul
        self.avg_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        
        self.max_mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=False),
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.avg_mlp(self.avg_pool(x))
        max_out = self.max_mlp(self.max_pool(x))
        
        attention = self.sigmoid(avg_out + max_out)
        
        return x * attention

class StairFeatureExtractor(nn.Module):
    """Extracteur de caractéristiques amélioré avec connexions résiduelles"""
    def __init__(self):
        super(StairFeatureExtractor, self).__init__()
        
        # Encoder path (downsampling) avec des connexions résiduelles
        self.enc1 = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64, use_residual=True)
        )
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(
            ConvBlock(64, 128),
            ConvBlock(128, 128, use_residual=True)
        )
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256, use_residual=True)
        )
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            ConvBlock(512, 512, use_residual=True)
        )
        self.pool4 = nn.MaxPool2d(2)
        
        # Bridge avec connexions résiduelles
        self.bridge = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512, use_residual=True)
        )
        
    def forward(self, x):
        # Encoder path avec skip connections
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

class MultiScaleFusion(nn.Module):
    """Module de fusion multi-échelle amélioré"""
    def __init__(self, channels):
        super(MultiScaleFusion, self).__init__()
        
        # Adaptations des dimensions pour chaque niveau
        self.adapt_bridge = nn.Conv2d(channels[0], channels[0], 1)
        self.adapt_e4 = nn.Conv2d(channels[1], channels[1], 1)
        self.adapt_e3 = nn.Conv2d(channels[2], channels[2], 1)
        self.adapt_e2 = nn.Conv2d(channels[3], channels[3], 1)
        self.adapt_e1 = nn.Conv2d(channels[4], channels[4], 1)
        
        # Pooling pour obtenir des caractéristiques globales
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, features):
        b, e4, e3, e2, e1 = features
        
        # Adapter chaque niveau
        b_adapted = self.adapt_bridge(b)
        e4_adapted = self.adapt_e4(e4)
        e3_adapted = self.adapt_e3(e3)
        e2_adapted = self.adapt_e2(e2)
        e1_adapted = self.adapt_e1(e1)
        
        # Appliquer le pooling global à chaque niveau
        b_feat = self.global_pool(b_adapted).flatten(1)
        e4_feat = self.global_pool(e4_adapted).flatten(1) 
        e3_feat = self.global_pool(e3_adapted).flatten(1)
        e2_feat = self.global_pool(e2_adapted).flatten(1)
        e1_feat = self.global_pool(e1_adapted).flatten(1)
        
        # Concaténer pour la fusion
        return torch.cat([b_feat, e4_feat, e3_feat, e2_feat, e1_feat], dim=1)

class StairNetDepth(nn.Module):
    """StairNetDepth amélioré avec fusion multi-échelle et attention"""
    def __init__(self, config=None):
        super(StairNetDepth, self).__init__()
        
        # Feature extractor backbone
        self.feature_extractor = StairFeatureExtractor()
        
        # Modules d'attention pour différentes échelles
        self.attention_bridge = AttentionModule(512)
        self.attention_e4 = AttentionModule(512)
        self.attention_e3 = AttentionModule(256)
        self.attention_e2 = AttentionModule(128)
        self.attention_e1 = AttentionModule(64)
        
        # Module de fusion multi-échelle
        self.fusion = MultiScaleFusion([512, 512, 256, 128, 64])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout adaptatif
        self.dropout = nn.Dropout(0.5)
        
        # Couches fully connected pour la régression avec dimensions adaptées
        # 512 + 512 + 256 + 128 + 64 = 1472
        self.fc1 = nn.Linear(1472, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)
        
        # ReLU séparés pour éviter inplace=True
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        
    def forward(self, x):
        # Extraire les caractéristiques multi-échelles
        features = self.feature_extractor(x)
        b, e4, e3, e2, e1 = features
        
        # Appliquer l'attention à chaque niveau en créant des copies pour éviter les modifications inplace
        b_att = self.attention_bridge(b.clone())
        e4_att = self.attention_e4(e4.clone())
        e3_att = self.attention_e3(e3.clone())
        e2_att = self.attention_e2(e2.clone())
        e1_att = self.attention_e1(e1.clone())
        
        # Fusion des caractéristiques multi-échelles
        fused_features = self.fusion((b_att, e4_att, e3_att, e2_att, e1_att))
        
        # Chemin de régression avec normalisations et activations
        x = self.fc1(fused_features)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.fc3(x)
        
        return x

# Mise à jour de la fonction get_model pour inclure le nouveau modèle
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