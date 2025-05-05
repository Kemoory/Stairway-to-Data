import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config_dl import BATCH_SIZE, DATA_TRANSFORMS

def load_data(data_path='data/stairsData_dump'):
    """Charger les chemins d'images et les labels depuis le dump de la base de données."""
    image_paths = []
    labels = []
    
    with open(data_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        if line.startswith("COPY public.images_data"):
            break
    
    for line in lines[lines.index(line) + 1:]:
        if line.strip() == '\\.':
            break
        parts = line.strip().split('\t')
        if len(parts) == 4:
            image_path = parts[1]
            label = int(parts[2])
            
            if image_path.startswith('.'):
                image_path = image_path[2:]
            
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(label)
            else:
                print(f"File not found: {image_path}")
    
    print(f"Collected {len(image_paths)} image paths and {len(labels)} labels")
    return image_paths, labels

class StairDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_loaders(image_paths, labels, kfold, batch_size=BATCH_SIZE):
    """Crée des loaders pour les données d'entrainement et de validation."""
    datasets = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths)):
        train_dataset = StairDataset(
            [image_paths[i] for i in train_idx],
            [labels[i] for i in train_idx],
            transform=DATA_TRANSFORMS
        )
        
        val_dataset = StairDataset(
            [image_paths[i] for i in val_idx],
            [labels[i] for i in val_idx],
            transform=DATA_TRANSFORMS
        )
        
        # Add drop_last=True to the training DataLoader
        datasets.append((
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),  # Modified line
            DataLoader(val_dataset, batch_size=batch_size)
        ))
    return datasets

