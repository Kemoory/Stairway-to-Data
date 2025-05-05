import os
import psycopg2
from config import db_config

def load_data(data_path=None):
    """Charger les chemins d'images et les labels depuis la base de données PostgreSQL."""
    image_paths = []
    labels = []
    
    try:
        # Connexion à la base de données
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # Exécution de la requête
        cursor.execute("""
            SELECT image_path, nombre_de_marches
            FROM public.images_data
            WHERE image_path IS NOT NULL
        """)
        
        # Récupération des résultats
        results = cursor.fetchall()
        
        # Traitement des résultats
        for row in results:
            image_path = row[0]
            label = row[1]
            
            # Ajustement du chemin si nécessaire
            if image_path.startswith('./'):
                image_path = image_path[2:]
            
            # Vérification de l'existence du fichier
            if os.path.isfile(image_path):
                image_paths.append(image_path)
                labels.append(label)
            else:
                print(f"File not found: {image_path}")
                
        cursor.close()
        conn.close()
        
        print(f"Collected {len(image_paths)} image paths and {len(labels)} labels from database")
        return image_paths, labels
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        return [], []