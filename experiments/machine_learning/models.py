import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from config import MODEL_PARAMS

class BaselineAverageRegressor(BaseEstimator, RegressorMixin):
    """A baseline model that always predicts the mean of the training data."""
    def __init__(self):
        self.mean_value = None
    
    def fit(self, X, y):
        """Store the mean of the training labels."""
        self.mean_value = np.mean(y)
        return self
    
    def predict(self, X):
        """Always return the mean value."""
        return np.full(len(X), self.mean_value)

def get_model(model_type):
    """Get initialized model based on type."""
    if model_type == 'random_forest':
        return RandomForestRegressor(**MODEL_PARAMS['random_forest'])
    elif model_type == 'svr':
        return SVR(**MODEL_PARAMS['svr'])
    elif model_type == 'gradient_boosting':
        return GradientBoostingRegressor(**MODEL_PARAMS['gradient_boosting'])
    elif model_type == 'kernel_ridge':
        return KernelRidge(alpha=1.0, kernel='rbf')
    elif model_type == 'baseline_average':
        return BaselineAverageRegressor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")