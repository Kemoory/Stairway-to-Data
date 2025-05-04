import os

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
    subset_size = int(len(image_paths) * 0.2)
    print(f"Collected {len(image_paths)} image paths and {len(labels)} labels")
    return image_paths, labels
    
    '''
    print(f"Subset size: {subset_size}")
    return image_paths[:subset_size], labels[:subset_size]
    '''
