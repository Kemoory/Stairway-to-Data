import os
from matplotlib.colors import LinearSegmentedColormap

# Chemins des répertoires
DATA_DIR = 'data'
MODEL_DIR = 'src/models'
RESULTS_DIR = 'results/visualisation/machine_learning'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Colormap personnalisée
CUSTOM_CMAP = LinearSegmentedColormap.from_list('custom_YlOrRd', 
                                            ['#ffffcc', '#ffeda0', '#fed976', '#feb24c', 
                                             '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026'])

# Paramètres des modèles mis à jour
MODEL_PARAMS = {
    'random_forest': {'n_estimators': 100, 'random_state': 42},
    'svr': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1},
    'gradient_boosting': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42},
    'kernel_ridge': {},
    'baseline_average': {},
    
    'polynomial_regression': {
        'degree': 2,
        'alpha': 1.0
    },
    'elastic_net': {
        'alpha': 1.0,
        'l1_ratio': 0.5
    },
    'xgboost': {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    },
    'advanced_svr': {
        'kernels': ['rbf', 'polynomial', 'sigmoid']
    }
}

# Paramètres d'évaluation
CV_FOLDS = 5
RANDOM_STATE = 42