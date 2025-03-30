import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import ElasticNet, Ridge
import xgboost as xgb

from config import MODEL_PARAMS
class BaselineAverageRegressor(BaseEstimator, RegressorMixin):
    """Modele de base qui predit toujours la moyenne des donnees d'entrainement."""
    def __init__(self):
        self.mean_value = None
    
    def fit(self, X, y):
        """Stockez la moyenne des libelles d'entrainement."""
        self.mean_value = np.mean(y)
        return self
    
    def predict(self, X):
        """Retournez toujours la valeur moyenne."""
        return np.full(len(X), self.mean_value)

def polynomial_regression_model(params=None):
    """Cree un mod le de Regression Polynomiale avec regularisation Ridge"""
    if params is None:
        params = MODEL_PARAMS.get('polynomial_regression', {})
    return make_pipeline(
        PolynomialFeatures(degree=params.get('degree', 2), include_bias=False),
        Ridge(alpha=params.get('alpha', 1.0), random_state=42)
    )

def elastic_net_model(params=None):
    """Cree un modele de Regression Elastic Net"""
    if params is None:
        params = MODEL_PARAMS.get('elastic_net', {})
    return ElasticNet(
        alpha=params.get('alpha', 1.0),
        l1_ratio=params.get('l1_ratio', 0.5),
        random_state=42,
        max_iter=1000
    )

def xgboost_model(params=None):
    """Cree un modele de Regression XGBoost"""
    if params is None:
        params = MODEL_PARAMS.get('xgboost', {})
    return xgb.XGBRegressor(
        n_estimators=params.get('n_estimators', 100),
        learning_rate=params.get('learning_rate', 0.1),
        max_depth=params.get('max_depth', 5),
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

def advanced_svr_models(params=None):
    """Cree plusieurs modeles SVR avec des noyaux differents"""
    if params is None:
        params = MODEL_PARAMS.get('advanced_svr', {})
    kernels = params.get('kernels', ['rbf', 'polynomial', 'sigmoid'])
    
    svr_models = {}
    for kernel in kernels:
        if kernel == 'rbf':
            svr_models[kernel] = make_pipeline(
                StandardScaler(),
                SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
            )
        elif kernel == 'polynomial':
            svr_models[kernel] = make_pipeline(
                StandardScaler(),
                SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1)
            )
        elif kernel == 'sigmoid':
            svr_models[kernel] = make_pipeline(
                StandardScaler(),
                SVR(kernel='sigmoid', C=1.0, gamma='scale')
            )
    
    return svr_models

def get_model(model_type):
    """Obtenir un modele initialiser en fonction de son type"""
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
    elif model_type == 'polynomial_regression':
        return polynomial_regression_model()
    elif model_type == 'elastic_net':
        return elastic_net_model()
    elif model_type == 'xgboost':
        return xgboost_model()
    elif model_type.startswith('advanced_svr_'):
        kernel = model_type.split('_')[-1]
        svr_models = advanced_svr_models()
        return svr_models.get(kernel)
    else:
        raise ValueError(f"Unknown model type: {model_type}")