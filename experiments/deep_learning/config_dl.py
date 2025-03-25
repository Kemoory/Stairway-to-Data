import os
import torch
from torchvision import transforms

# Chemins des répertoires
DATA_DIR = 'data'
MODEL_DIR = 'src/models'
RESULTS_DIR = 'results/deep_learning'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Paramètres d'entraînement
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
CV_FOLDS = 3
RANDOM_STATE = 42
IMAGE_SIZE = 224

# Augmentation des données
DATA_TRANSFORMS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Configuration des modèles
MODEL_CONFIGS = {
    'resnet18': {'pretrained': True, 'freeze_backbone': False},
    'simple_cnn': {'channels': [32, 64, 128], 'dropout': 0.2},
    'vit': {'pretrained': True, 'image_size': IMAGE_SIZE}
}