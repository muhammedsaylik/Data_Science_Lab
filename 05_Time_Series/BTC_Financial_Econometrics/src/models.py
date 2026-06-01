"""
Non-Parametric Time Series Models for Bitcoin Forecasting
Designed for Walk-Forward Validation & Regime Switching
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class DirectionalForecaster(BaseEstimator, ClassifierMixin):
    """
    Base class for time-series directional forecasting (up/down binary classification).
    Implements walk-forward cross-validation and regime detection.
    """
    
    def __init__(self, model_type='xgboost', **kwargs):
        """
        Parameters
        ----------
        model_type : str
            'xgboost', 'lightgbm', or 'rf'
        **kwargs : dict
            Model hyperparameters
        """
        self.model_type = model_type
        self.model = None
        self.params = kwargs
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize underlying model based on type"""
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            # Merge defaults with user params (user params override defaults)
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42,
                'n_jobs': -1,
            }
            xgb_params.update(self.params)
            self.model = XGBClassifier(**xgb_params)
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            # Merge defaults with user params
            lgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_samples': 10,
                'num_leaves': 31,
                'random_state': 42,
                'n_jobs': -1,
            }
            lgb_params.update(self.params)
            self.model = LGBMClassifier(**lgb_params)
        else:
            # Fallback to RandomForest
            rf_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'min_samples_split': 10,
                'min_samples_leaf': 5,
                'random_state': 42,
                'n_jobs': -1,
            }
            rf_params.update(self.params)
            self.model = RandomForestClassifier(**rf_params)
    
    def fit(self, X, y):
        """Fit model to training data"""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Predict binary direction (0=down, 1=up)"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probability of upward movement"""
        return self.model.predict_proba(X)
    
    def feature_importance(self):
        """Extract feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None


class AdaptiveEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble method that adapts weights based on market regime.
    Uses volatility clustering to detect regime changes.
    """
    
    def __init__(self, vol_threshold=0.03):
        """
        Parameters
        ----------
        vol_threshold : float
            Volatility threshold for regime switching (annualized)
        """
        self.vol_threshold = vol_threshold
        self.models = {
            'trending': DirectionalForecaster('xgboost', max_depth=8) if XGBOOST_AVAILABLE else RandomForestClassifier(max_depth=8),
            'ranging': DirectionalForecaster('lightgbm', max_depth=4) if LIGHTGBM_AVAILABLE else RandomForestClassifier(max_depth=4),
        }
        self.regimes = None
        self.weights = {'trending': 0.5, 'ranging': 0.5}
    
    def _detect_regimes(self, X, vol_col_idx=None):
        """Detect market regimes based on volatility"""
        # Assume first column is returns or volatility-related
        if vol_col_idx is None:
            vol_col_idx = 0
        
        vol = X[:, vol_col_idx].std()
        regimes = np.where(vol > self.vol_threshold, 'trending', 'ranging')
        return regimes
    
    def fit(self, X, y):
        """Fit ensemble on entire dataset"""
        self.regimes = self._detect_regimes(X)
        
        # Fit trending model
        trending_mask = self.regimes == 'trending'
        if trending_mask.sum() > 10:
            self.models['trending'].fit(X[trending_mask], y[trending_mask])
        
        # Fit ranging model
        ranging_mask = self.regimes == 'ranging'
        if ranging_mask.sum() > 10:
            self.models['ranging'].fit(X[ranging_mask], y[ranging_mask])
        
        return self
    
    def predict(self, X):
        """Predict with regime-aware ensemble"""
        regimes = self._detect_regimes(X)
        predictions = np.zeros(len(X), dtype=int)
        
        for regime in ['trending', 'ranging']:
            mask = regimes == regime
            if mask.sum() > 0:
                predictions[mask] = self.models[regime].predict(X[mask])
        
        return predictions
    
    def predict_proba(self, X):
        """Probability predictions with regime weighting"""
        regimes = self._detect_regimes(X)
        proba = np.zeros((len(X), 2))
        
        for regime in ['trending', 'ranging']:
            mask = regimes == regime
            if mask.sum() > 0:
                regime_pred = self.models[regime].predict_proba(X[mask])
                weight = self.weights[regime]
                proba[mask] += weight * regime_pred
        
        # Normalize
        proba = proba / proba.sum(axis=1, keepdims=True)
        return proba


class WalkForwardValidator:
    """
    Implements walk-forward validation for time-series models.
    Ensures no look-ahead bias and realistic out-of-sample evaluation.
    """
    
    def __init__(self, n_splits=10, test_size=None, gap=0):
        """
        Parameters
        ----------
        n_splits : int
            Number of walk-forward splits
        test_size : int, optional
            Size of test fold (default: n_samples // n_splits)
        gap : int
            Gap between train and test (prevents data leakage)
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.splits = []
        self.scores = []
    
    def split(self, X, y=None):
        """Generate walk-forward train/test indices"""
        n_samples = len(X)
        test_size = self.test_size or (n_samples // self.n_splits)
        
        for i in range(self.n_splits):
            train_end = i * test_size
            test_start = train_end + self.gap
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def evaluate(self, model, X, y, metric='balanced_accuracy'):
        """
        Evaluate model using walk-forward validation.
        
        Parameters
        ----------
        model : estimator
            Fitted model
        X : array-like
            Features
        y : array-like
            Targets
        metric : str
            Evaluation metric ('balanced_accuracy', 'f1', 'roi')
        
        Returns
        -------
        scores : list
            Out-of-sample scores for each fold
        """
        from sklearn.metrics import balanced_accuracy_score, f1_score
        
        scores = []
        for train_idx, test_idx in self.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            if metric == 'balanced_accuracy':
                score = balanced_accuracy_score(y_test, y_pred)
            elif metric == 'f1':
                score = f1_score(y_test, y_pred, average='weighted')
            elif metric == 'roi':
                # Simplified ROI: +1 if correct prediction, -1 if wrong
                score = ((y_pred == y_test) * 2 - 1).mean()
            else:
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_test, y_pred)
            
            scores.append(score)
        
        self.scores = scores
        return scores
    
    @property
    def mean_score(self):
        """Mean out-of-sample score"""
        return np.mean(self.scores) if self.scores else None
    
    @property
    def std_score(self):
        """Standard deviation of scores"""
        return np.std(self.scores) if self.scores else None
