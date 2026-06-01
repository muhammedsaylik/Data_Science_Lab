import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class DataCleaner(BaseEstimator, TransformerMixin):
    """Veri temizliği yapar"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Inf değerleri NaN'a çevir
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df


class MissingValueHandler(BaseEstimator, TransformerMixin):
    """Eksik değerleri doldurur"""
    
    def __init__(self, strategy='forward_fill'):
        self.strategy = strategy
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if self.strategy == 'forward_fill':
            df = df.ffill()
            df = df.bfill()
        elif self.strategy == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
                df[col] = df[col].ffill()
                df[col] = df[col].bfill()
        
        return df


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Aykırı değerleri işler"""
    
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.means_ = None
        self.stds_ = None
    
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.means_ = X[numeric_cols].mean()
        self.stds_ = X[numeric_cols].std()
        return self
    
    def transform(self, X):
        df = X.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            z_scores = np.abs((df[col] - self.means_[col]) / (self.stds_[col] + 1e-10))
            df.loc[z_scores > self.threshold, col] = np.nan
        
        return df


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Özellikleri normalize eder"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.numeric_cols_ = None
    
    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        self.numeric_cols_ = numeric_cols
        self.scaler.fit(X[numeric_cols])
        return self
    
    def transform(self, X):
        df = X.copy()
        df[self.numeric_cols_] = self.scaler.transform(df[self.numeric_cols_])
        return df


def create_preprocessing_pipeline():
    """Veri ön işleme pipeline'ını oluşturur"""
    pipeline = Pipeline([
        ('cleaner', DataCleaner()),
        ('outlier_handler', OutlierHandler(threshold=3)),
        ('missing_handler', MissingValueHandler(strategy='interpolate')),
        ('scaler', FeatureScaler())
    ])
    return pipeline


def create_feature_engineering_pipeline():
    """Feature engineering pipeline'ını oluşturur"""
    from .features import (
        TechnicalFeatures, VolatilityFeatures, MicrostructureFeatures,
        MomentumFeatures, TrendFeatures, CycleFeatures, ComplexFeatures,
        VolatilityRiskPremium, SentimentFeatures
    )
    
    pipeline = Pipeline([
        ('technical', TechnicalFeatures()),
        ('volatility', VolatilityFeatures()),
        ('microstructure', MicrostructureFeatures()),
        ('momentum', MomentumFeatures()),
        ('trend', TrendFeatures()),
        ('cycle', CycleFeatures()),
        ('complex', ComplexFeatures()),
        ('vol_risk_premium', VolatilityRiskPremium()),
        ('sentiment', SentimentFeatures()),
    ])
    return pipeline
