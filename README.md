# Stairway-to-heaven

__Detection et Comptage de Marches d'Escalier__

## Description
Ce projet a pour objectif de détecter et de compter automatiquement le nombre de marches d'un escalier à partir d'une image capturée par un téléphone. L'image fournie est centrée sur les marches et possède un fond relativement homogène.

## Objectifs
- **Acquisition des images** : Construire un jeu de données avec des annotations manuelles (vérité terrain).
- **Détection et comptage des marches** : Développer une méthode permettant d'identifier et de compter les marches.
- **Évaluation des performances** : Comparer les résultats obtenus avec la vérité terrain.
- **Proposition d'améliorations** : Analyser les résultats et suggérer des optimisations.

## Arborescence du projet
```
stair_detection/
│── data/                    # Dossier contenant les images et annotations
│   ├── raw/                 # Images brutes capturées
│   ├── processed/           # Images prétraitées
│   ├── labels.json          # Vérité terrain (nombre de marches par image)
│
│── src/                     # Code source principal
│   ├── preprocessing.py      # Prétraitement des images (filtrage, seuillage, etc.)
│   ├── detection.py          # Algorithme de détection et comptage des marches
│   ├── evaluation.py         # Évaluation des performances du modèle
│   ├── visualization.py      # Affichage des résultats et overlays
│   ├── utils.py              # Fonctions utilitaires
│
│── tests/                    # Tests unitaires
│   ├── test_detection.py
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│
│── results/                  # Dossier pour les résultats des expériences
│
│── main.py                   # Script principal pour exécuter le pipeline
│── requirements.txt           # Dépendances Python
│── README.md                  # Explication du projet et de son exécution
│── report.pdf                 # Rapport du projet (optionnel)
│── presentation.pptx          # Présentation finale
```

## Installation
1. Cloner le projet :
```bash
git clone https://github.com/Kemoory/Stairway-to-heaven.git
cd Stairway-to-heaven
```
2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation
Pour exécuter la détection sur une image donnée :
```bash
python main.py --image data/raw/XXX.jpg
```

## Dépendances
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib
- Scikit-image
- SciPy

## Évaluation
L'évaluation des performances est réalisée en comparant les résultats obtenus avec la vérité terrain. Les métriques utilisées incluent :
- **Erreur absolue moyenne**
- **Précision / Rappel / F1-score**

## Contributions
Les contributions sont les bienvenues ! Merci de suivre ces étapes :
1. Forker le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature-nouvelle-fonctionnalite`)
3. Committer vos modifications (`git commit -m "Ajout d'une nouvelle fonctionnalité"`)
4. Pusher la branche (`git push origin feature-nouvelle-fonctionnalite`)
5. Ouvrir une pull request

## Licence
Ce projet est sous n'as pas encore de licence.