from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from config import MODEL_PARAMS

def get_model(model_type):
    """Get initialized model based on type."""
    if model_type == 'random_forest':
        return RandomForestRegressor(**MODEL_PARAMS['random_forest'])
    elif model_type == 'svr':
        return SVR(**MODEL_PARAMS['svr'])
    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(**MODEL_PARAMS['gradient_boosting'])
    elif model_type == 'mlp':
        return MLPRegressor(**MODEL_PARAMS['mlp'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")