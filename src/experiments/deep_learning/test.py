import torch
from models import StairNetDepth
from config_dl import IMAGE_SIZE
import time

def test_stairnet_model():
    """Test le modèle StairNet-Depth avec une entrée aléatoire."""
    print("=== Test du modèle StairNet-Depth ===")
    
    # Création d'une configuration minimale
    config = {'pretrained': True}
    
    # Initialisation du modèle
    try:
        model = StairNetDepth(config)
        print("✓ Modèle initialisé avec succès")
    except Exception as e:
        print(f"✗ Erreur lors de l'initialisation du modèle: {str(e)}")
        return False
    
    # Vérification des composants principaux
    components = [
        ('feature_extractor', model.feature_extractor),
        ('global_pool', model.global_pool),
        ('attention1', model.attention1),
        ('attention2', model.attention2),
        ('fc1', model.fc1),
        ('fc2', model.fc2),
        ('fc3', model.fc3)
    ]
    
    for name, component in components:
        if component is None:
            print(f"✗ Composant {name} est None")
            return False
        else:
            print(f"✓ Composant {name} correctement initialisé")
    
    # Test avec une entrée factice
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation du périphérique: {device}")
    
    model = model.to(device)
    dummy_input = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    
    try:
        start_time = time.time()
        output = model(dummy_input)
        inference_time = time.time() - start_time
        
        print(f"✓ Forward pass réussi en {inference_time:.4f} secondes")
        print(f"✓ Forme de sortie: {output.shape}")
        print(f"✓ Exemple de valeurs: {output.detach().cpu().numpy()}")
        return True
    except Exception as e:
        print(f"✗ Erreur lors du forward pass: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_stairnet_model()
    if success:
        print("\nLe modèle StairNet-Depth est prêt à être utilisé!")
    else:
        print("\nDes corrections sont nécessaires avant d'utiliser le modèle.")