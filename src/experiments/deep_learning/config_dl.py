import os
import torch
from torchvision import transforms
from matplotlib.colors import LinearSegmentedColormap

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
'''
#paramètres de TEST
BATCH_SIZE = 8  # Réduit la consommation mémoire
EPOCHS = 2  # Seulement 2 époques pour le test
LEARNING_RATE = 1e-3  # Garder identique
CV_FOLDS = 2  # Moins de validation croisée
RANDOM_STATE = 42
IMAGE_SIZE = 128  # Images plus petites
'''

# Colormap personnalisée
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_YlOrRd', 
                                            ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', 
                                             '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'])

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
    'vit': {'pretrained': True, 'image_size': IMAGE_SIZE},
    'stairnet_depth': {'pretrained': True}
}

db_config={
    'dbname': 'stairDatas',
    'user': 'zack',
    'password': 'zack',
    'host': 'localhost',
    'port': '5432'
}