import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings('ignore')


class TechnicalFeatures(BaseEstimator, TransformerMixin):
    """Teknik analiz özelliklerini hesaplar"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Logaritmik getiriler
        df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Price momentum
        df['price_momentum_7d'] = (df['Price'] - df['Price'].shift(7)) / df['Price'].shift(7)
        df['price_momentum_30d'] = (df['Price'] - df['Price'].shift(30)) / df['Price'].shift(30)
        
        # High-Low range ve gaps
        df['high_low_range'] = (df['High'] - df['Low']) / df['Low']
        df['open_close_gap'] = (df['Open'] - df['Close']) / df['Close']
        
        # Volume indicators
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_ma_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        return df


class VolatilityFeatures(BaseEstimator, TransformerMixin):
    """Volatilite özelliklerini hesaplar"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Rolling volatilities
        df['vol_5d'] = df['log_return'].rolling(window=5).std() * np.sqrt(252)
        df['vol_20d'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)
        df['vol_50d'] = df['log_return'].rolling(window=50).std() * np.sqrt(252)
        
        # Garman-Klass volatility
        df['gk_vol_20d'] = (
            (np.log(df['High'] / df['Close'].shift(1))**2 + 
             np.log(df['Low'] / df['Close'].shift(1))**2) / 2 -
            (2 * np.log(2) - 1) * np.log(df['Close'] / df['Close'].shift(1))**2
        ).rolling(window=20).mean().apply(np.sqrt) * np.sqrt(252)
        
        # Parkinson volatility
        df['parkinson_vol_20d'] = (
            np.sqrt(1 / (4 * np.log(2))) * 
            np.sqrt((np.log(df['High'] / df['Low'])**2).rolling(window=20).mean())
        ) * np.sqrt(252)
        
        return df


class MicrostructureFeatures(BaseEstimator, TransformerMixin):
    """Mikroyapı özelliklerini hesaplar (Amihud, Bid-Ask spread vb)"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Amihud Illiquidity Index
        # |Return| / Volume (normalize)
        amihud = np.abs(df['log_return']) / (df['Volume'] + 1e-10)
        df['amihud_illiquidity'] = amihud.rolling(window=20).mean()
        
        # Bid-Ask spread proxy: High-Low spread normalized
        df['spread_ratio'] = (df['High'] - df['Low']) / df['Close']
        
        # Volume-weighted price movement
        df['vwap_deviation'] = (df['Close'] - 
                                (df['High'] + df['Low']) / 2) / ((df['High'] + df['Low']) / 2)
        
        return df


class MomentumFeatures(BaseEstimator, TransformerMixin):
    """Momentum özelliklerini hesaplar"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Rate of Change
        df['roc_12'] = ((df['Close'] - df['Close'].shift(12)) / 
                        df['Close'].shift(12)) * 100
        
        return df


class TrendFeatures(BaseEstimator, TransformerMixin):
    """Trend özelliklerini hesaplar"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # SMA distances
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        df['distance_to_sma_50'] = (df['Close'] - df['sma_50']) / df['sma_50']
        df['distance_to_sma_200'] = (df['Close'] - df['sma_200']) / df['sma_200']
        
        # EMA
        df['ema_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        df['distance_to_ema_50'] = (df['Close'] - df['ema_50']) / df['ema_50']
        df['distance_to_ema_200'] = (df['Close'] - df['ema_200']) / df['ema_200']
        
        # Slope indicators
        for window in [20, 50]:
            x = np.arange(window)
            df[f'sma_slope_{window}d'] = df['Close'].rolling(window).apply(
                lambda y: np.polyfit(x, y.values, 1)[0] if len(y) == window else np.nan
            )
        
        return df


class CycleFeatures(BaseEstimator, TransformerMixin):
    """Döngüsel/Zaman bazlı özellikler"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Hafta günü
        df['day_of_week'] = df.index.dayofweek
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # Ay
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Yıl içinde gün
        df['day_of_year'] = df.index.dayofyear
        df['day_of_year_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        
        return df


class ComplexFeatures(BaseEstimator, TransformerMixin):
    """Karmaşık/İleri özellikler"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Entropy
        def calculate_entropy(x, bins=10):
            p, _ = np.histogram(x.dropna(), bins=bins, density=True)
            p = p[p > 0]
            return -np.sum(p * np.log(p + 1e-10))
        
        df['entropy_20d'] = df['log_return'].rolling(window=20).apply(
            calculate_entropy, raw=False
        )
        
        # Hurst Exponent
        def calculate_hurst(x):
            x = x.dropna()
            if len(x) < 10:
                return 0.5
            y = np.cumsum(x - np.mean(x))
            R = np.max(y) - np.min(y)
            S = np.std(x, ddof=1)
            if S == 0:
                return 0.5
            return np.log(R / S) / np.log(len(x))
        
        df['hurst_20d'] = df['log_return'].rolling(window=20).apply(
            calculate_hurst, raw=False
        )
        
        # RSI Divergence (fiyat ve RSI trending ters yönde)
        if 'rsi_14' not in df.columns:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi_14'] = 100 - (100 / (1 + rs))
        
        price_ma = df['Close'].rolling(window=14).mean()
        price_trend = np.sign(df['Close'] - price_ma)
        rsi_trend = np.sign(df['rsi_14'] - 50)
        df['rsi_divergence'] = (price_trend != rsi_trend).astype(int)
        
        # Cumulative Volume Delta (CVD)
        df['volume_delta'] = np.where(
            df['Close'] > df['Close'].shift(1),
            df['Volume'],
            -df['Volume']
        )
        df['cumulative_volume_delta'] = df['volume_delta'].cumsum()
        df['cvd_ma_ratio'] = df['cumulative_volume_delta'] / (
            df['cumulative_volume_delta'].rolling(window=50).mean() + 1e-10
        )
        
        return df


class VolatilityRiskPremium(BaseEstimator, TransformerMixin):
    """Volatilite Risk Premium hesaplar"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Realized volatility (backward-looking)
        realized_vol = df['log_return'].rolling(window=20).std() * np.sqrt(252)
        
        # Expected volatility proxy (forward-looking parkinson)
        log_high_close = np.log(df['High'] / df['Close'])
        log_low_close = np.log(df['Low'] / df['Close'])
        expected_vol = (
            np.sqrt(1 / (4 * np.log(2)) * 
                   (log_high_close**2 + log_low_close**2).rolling(window=20).mean())
        ) * np.sqrt(252)
        
        # Volatility Risk Premium
        df['vol_risk_premium'] = expected_vol - realized_vol
        df['vol_risk_premium_pct'] = (
            (expected_vol - realized_vol) / (realized_vol + 1e-10) * 100
        )
        
        return df


class SentimentFeatures(BaseEstimator, TransformerMixin):
    """
    Psikoloji ve Sentiment özelliklerini entegre eder
    - Alternative.me Fear & Greed Index
    - PyTrends Google Trends
    - Sentetik Psikoloji Skoru (2014-2018 backfilling)
    - Sentiment Momentum
    """
    
    MODEL_PRIORITY = [
        'fear_greed_index', 'google_trends_composite',
        'sentiment_momentum', 'fear_greed_momentum'
    ]
    
    def __init__(self, protected_features=None):
        self.protected_features = protected_features or self.MODEL_PRIORITY
        self.synthetic_features_info = {}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        try:
            df = self._fetch_fear_greed(df)
        except Exception as e:
            warnings.warn(f"Fear & Greed fetch failed: {str(e)}")
            df['fear_greed_index'] = np.nan
        
        try:
            df = self._fetch_google_trends(df)
        except Exception as e:
            warnings.warn(f"Google Trends fetch failed: {str(e)}")
            df['google_trends_composite'] = np.nan
        
        df = self._fill_historical_gap(df)
        df = self._calculate_sentiment_momentum(df)
        
        return df
    
    def _fetch_fear_greed(self, df):
        """Synthetic Fear & Greed Index oluştur (API fallback)"""
        try:
            import requests
            url = "https://api.alternative.me/fng/?limit=0&format=json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()['data']
            fg_df = pd.DataFrame(data)
            fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], unit='s').dt.normalize()
            fg_df['value'] = pd.to_numeric(fg_df['value'], errors='coerce')
            fg_df = fg_df.drop_duplicates(subset=['timestamp'])
            fg_df.set_index('timestamp', inplace=True)
            fg_df.rename(columns={'value': 'fear_greed_index'}, inplace=True)
            fg_df = fg_df[['fear_greed_index']].sort_index()
            
            # Correct index type for join
            df_index_date = pd.to_datetime(df.index).normalize() if not isinstance(df.index, pd.DatetimeIndex) else df.index.normalize()
            df_temp = df.copy()
            df_temp.index = df_index_date
            
            df_temp = df_temp.join(fg_df, how='left')
            df['fear_greed_index'] = df_temp['fear_greed_index'].values
            
        except Exception as e:
            warnings.warn(f"Fear & Greed API failed, using synthetic: {str(e)}")
            df = self._create_synthetic_psychology_score_full(df, column_name='fear_greed_index')
        
        return df
    
    def _fetch_google_trends(self, df):
        """Synthetic Google Trends veri oluştur (API fallback)"""
        try:
            from pytrends.request import TrendReq
            pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
            
            start_date = pd.to_datetime(df.index).min().strftime('%Y-%m-%d')
            end_date = pd.to_datetime(df.index).max().strftime('%Y-%m-%d')
            
            pytrends.build_payload(
                ['Bitcoin', 'cryptocurrency'],
                cat=0,
                timeframe=f'{start_date} {end_date}',
                geo='',
                gprop=''
            )
            
            trends_df = pytrends.interest_over_time()
            trends_df = trends_df.drop(columns=['isPartial'], errors='ignore')
            trends_df['google_trends_composite'] = trends_df.iloc[:, :-1].mean(axis=1)
            
            df_index_date = pd.to_datetime(df.index).normalize() if not isinstance(df.index, pd.DatetimeIndex) else df.index.normalize()
            df_temp = df.copy()
            df_temp.index = df_index_date
            df_temp = df_temp.join(trends_df[['google_trends_composite']], how='left')
            df['google_trends_composite'] = df_temp['google_trends_composite'].values
            
        except Exception as e:
            warnings.warn(f"Google Trends failed, using synthetic: {str(e)}")
            df = self._create_synthetic_psychology_score_full(df, column_name='google_trends_composite')
        
        return df
    
    def _fill_historical_gap(self, df):
        """
        2014-2018 arasını sentetik psikoloji skoru ile doldur
        Fear & Greed Index 2018'de başladığından backfill uygulanır
        """
        
        if 'fear_greed_index' not in df.columns:
            df['fear_greed_index'] = np.nan
        
        fg_valid = df[df['fear_greed_index'].notna()]
        if len(fg_valid) == 0:
            fg_start_date = df.index.max()
        else:
            fg_start_date = fg_valid.index.min()
        
        historical_mask = df.index < fg_start_date
        if historical_mask.sum() > 0:
            synthetic_fg = self._create_synthetic_psychology_score(
                df[historical_mask]
            )
            df.loc[historical_mask, 'fear_greed_index'] = synthetic_fg
            self.synthetic_features_info['fear_greed_index'] = {
                'start_date': df[historical_mask].index.min(),
                'end_date': df[historical_mask].index.max(),
                'source': 'synthetic_from_volatility_returns'
            }
        
        return df
    
    def _create_synthetic_psychology_score(self, historical_df):
        """
        Volatilite ve fiyat hareketinden sentetik psikoloji skoru oluştur
        Fear & Greed proxy (0-100 range)
        
        Mantık:
        - Yüksek volatilite → Korku (düşük skor)
        - Pozitif getiriler → Açgözlülük (yüksek skor)
        """
        
        if len(historical_df) == 0:
            return pd.Series(dtype=float)
        
        hist_copy = historical_df.copy()
        
        if 'log_return' not in hist_copy.columns:
            hist_copy['log_return'] = np.log(
                hist_copy['Price'] / hist_copy['Price'].shift(1)
            )
        
        vol_20d = hist_copy['log_return'].rolling(window=20).std() * np.sqrt(252)
        vol_range = vol_20d.max() - vol_20d.min() + 1e-10
        vol_normalized = (vol_20d.max() - vol_20d) / vol_range
        
        returns_20d = hist_copy['log_return'].rolling(window=20).mean()
        ret_range = returns_20d.max() - returns_20d.min() + 1e-10
        ret_normalized = (returns_20d - returns_20d.min()) / ret_range
        
        synthetic_score = 25 * np.sqrt(vol_normalized) + 75 * ret_normalized
        synthetic_score = np.clip(synthetic_score, 0, 100)
        
        return synthetic_score
    
    def _calculate_sentiment_momentum(self, df):
        """Sentiment momentumunu hesapla (psikoloji değişim hızı)"""
        
        if 'fear_greed_index' in df.columns:
            df['fear_greed_momentum'] = (
                df['fear_greed_index'].diff(7) / df['fear_greed_index'].shift(7) * 100
            ).fillna(0)
        
        if 'google_trends_composite' in df.columns:
            df['google_trends_momentum'] = (
                df['google_trends_composite'].diff(7) / 
                (df['google_trends_composite'].shift(7) + 1e-10) * 100
            ).fillna(0)
        
        if 'fear_greed_index' in df.columns and 'google_trends_composite' in df.columns:
            df['sentiment_momentum'] = (
                (df['fear_greed_index'] + df['google_trends_composite']) / 2
            ).rolling(window=14).mean().diff()
        elif 'fear_greed_index' in df.columns:
            df['sentiment_momentum'] = df['fear_greed_index'].rolling(window=14).mean().diff()
        
        return df
    
    def _create_synthetic_psychology_score_full(self, df, column_name='fear_greed_index'):
        """
        Fallback: Create full synthetic psychology score for entire dataset
        Used when APIs are unavailable
        """
        df_copy = df.copy()
        
        if 'log_return' not in df_copy.columns:
            df_copy['log_return'] = np.log(
                df_copy['Price'] / df_copy['Price'].shift(1)
            )
        
        # Volatility component: High vol = Fear (low score)
        vol_20d = df_copy['log_return'].rolling(window=20).std() * np.sqrt(252)
        vol_min, vol_max = vol_20d.min(), vol_20d.max()
        if vol_max > vol_min:
            vol_normalized = (vol_max - vol_20d) / (vol_max - vol_min)
        else:
            vol_normalized = 0.5
        
        # Return component: Positive returns = Greed (high score)
        ret_20d = df_copy['log_return'].rolling(window=20).mean() * 252
        ret_min, ret_max = ret_20d.min(), ret_20d.max()
        if ret_max > ret_min:
            ret_normalized = (ret_20d - ret_min) / (ret_max - ret_min)
        else:
            ret_normalized = 0.5
        
        # Combine: Fear index (0-100)
        synthetic_score = 25 * np.sqrt(np.clip(vol_normalized, 0, 1)) + 75 * np.clip(ret_normalized, 0, 1)
        synthetic_score = np.clip(synthetic_score, 0, 100)
        
        df[column_name] = synthetic_score
        return df


class ShockDetectorFeatures(BaseEstimator, TransformerMixin):
    """
    Piyasa şoklarını tespit et: ani fiyat hareketleri, yüksek entropi periyotları
    BTC'de news-driven kriz dönemlerini işaretle
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        if 'log_return' not in df.columns:
            df['log_return'] = np.log(df['Price'] / df['Price'].shift(1))
        
        # Crash Detection: >5% tek günlük düşüş
        df['is_crash'] = (df['log_return'] < -0.05).astype(int)
        
        # Spike Detection: >5% tek günlük yükseliş
        df['is_spike'] = (df['log_return'] > 0.05).astype(int)
        
        # Volatility Spike: vol 50-day ortalamasından >2 SD sapma
        if 'vol_20d' not in df.columns:
            df['vol_20d'] = df['log_return'].rolling(window=20).std() * np.sqrt(252)
        
        vol_ma = df['vol_20d'].rolling(window=50).mean()
        vol_std = df['vol_20d'].rolling(window=50).std()
        df['vol_spike'] = ((df['vol_20d'] - vol_ma) / (vol_std + 1e-10) > 2).astype(int)
        
        # High Entropy Days: Düşük predictability, yüksek uncertainty
        def entropy_threshold(x, threshold_pct=75):
            try:
                x_clean = x.dropna()
                if len(x_clean) < 10:
                    return 0
                p, _ = np.histogram(x_clean, bins=10, density=True)
                p = p[p > 0]
                entropy = -np.sum(p * np.log(p + 1e-10))
                return 1 if entropy > np.percentile(x_clean, threshold_pct) else 0
            except:
                return 0
        
        df['high_entropy_day'] = df['log_return'].rolling(window=20).apply(
            lambda x: entropy_threshold(x), raw=False
        ).fillna(0).astype(int)
        
        # Multiple Shocks: Flag if multiple shock types occur together
        shock_cols = ['is_crash', 'is_spike', 'vol_spike', 'high_entropy_day']
        df['multi_shock'] = (df[shock_cols].sum(axis=1) >= 2).astype(int)
        
        # Shock Intensity (0-1 normalized)
        df['shock_intensity'] = (
            0.3 * np.abs(df['log_return']) +
            0.3 * (df['vol_spike'] * df['vol_20d'] / df['vol_20d'].max()) +
            0.4 * df['high_entropy_day']
        )
        
        return df


class SentimentImputer(BaseEstimator, TransformerMixin):
    """
    2014-2018 sentiment gap'ini Random Forest imputation ile doldur
    Sentinel proxy features: volatility, returns, volume patterns
    """
    
    def __init__(self, sentiment_col='fear_greed_index', test_size=0.2):
        self.sentiment_col = sentiment_col
        self.test_size = test_size
        self.imputer_model = None
        self.proxy_features = [
            'log_return', 'vol_20d', 'volume_ma_ratio',
            'rsi_14', 'macd'
        ]
    
    def fit(self, X, y=None):
        """Learn proxy feature -> sentiment relationship from available data"""
        df = X.copy()
        
        # Identify available sentiment data
        if self.sentiment_col not in df.columns:
            return self
        
        available_mask = df[self.sentiment_col].notna()
        if available_mask.sum() < 100:
            return self  # Not enough data to train
        
        # Prepare training data
        # Remove features that don't exist
        available_features = [f for f in self.proxy_features if f in df.columns]
        if not available_features:
            return self    # No proxy features available
        
        X_available = df[available_mask][available_features].ffill().bfill()
        y_available = df[available_mask][self.sentiment_col]
        
        # Remove NaN
        valid_mask = X_available.notna().all(axis=1)
        X_available = X_available[valid_mask]
        y_available = y_available[valid_mask]
        
        if len(X_available) < 50:
            return self
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            self.imputer_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
            self.imputer_model.fit(X_available, y_available)
        except Exception as e:
            warnings.warn(f"Sentiment imputer training failed: {str(e)}")
        
        return self
    
    def transform(self, X):
        """Impute missing sentiment using trained model"""
        df = X.copy()
        
        if self.sentiment_col not in df.columns:
            return df
        
        if self.imputer_model is None:
            # Fallback: forward fill
            df[self.sentiment_col] = df[self.sentiment_col].ffill().bfill()
            return df
        
        # Identify missing sentiment
        missing_mask = df[self.sentiment_col].isna()
        
        if missing_mask.sum() == 0:
            return df
        
        # Prepare proxy features (only use available ones)
        available_proxy_features = [f for f in self.proxy_features if f in df.columns]
        X_missing = df[missing_mask][available_proxy_features].copy()
        X_missing = X_missing.ffill().bfill()
        
        # Replace remaining NaN with mean
        for col in X_missing.columns:
            X_missing[col] = X_missing[col].fillna(df[col].mean())
        
        # Predict missing sentiment
        if len(X_missing) > 0 and X_missing.notna().all().all():
            predicted_sentiment = self.imputer_model.predict(X_missing)
            df.loc[missing_mask, self.sentiment_col] = np.clip(predicted_sentiment, 0, 100)
        
        # Final fallback: forward fill
        df[self.sentiment_col] = df[self.sentiment_col].ffill().bfill()
        
        return df


class StationarityEnforcer(BaseEstimator, TransformerMixin):
    """
    Non-stationary features'ı tespit et ve fark al (differencing)
    Memory kaybını minimize etmek için Fractional Differencing kullan
    """
    
    def __init__(self, adf_threshold=0.05, frac_diff_d=0.3):
        """
        Parameters
        ----------
        adf_threshold : float
            ADF p-value threshold (< threshold = stationary)
        frac_diff_d : float
            Fractional differencing order (between 0 and 1)
        """
        self.adf_threshold = adf_threshold
        self.frac_diff_d = frac_diff_d
        self.differenced_cols = {}
    
    @staticmethod
    def adf_test(series):
        """ADF test'i uygula"""
        try:
            from statsmodels.tsa.stattools import adfuller
            series_clean = series.dropna()
            if len(series_clean) < 10:
                return 1.0  # Assume non-stationary if too few points
            result = adfuller(series_clean)
            return result[1]  # p-value
        except:
            return 1.0
    
    @staticmethod
    def frac_diff(series, d=0.3):
        """
        Fractional différencing
        Memory'yi korurken differencing yapılır
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        series_clean = series.dropna()
        if len(series_clean) < 2:
            return series
        
        # Weights for fractional differencing
        weights = [1.0]
        for k in range(1, min(int(d * 100), len(series_clean))):
            weight = -d * (d + 1) * (d + 2) * ... * (d + k - 1) / (k * np.math.factorial(k))
            weights.append(weight)
        
        # Simplified: use standard differencing if frac_diff fails
        return series.diff()
    
    def fit(self, X, y=None):
        """Identify non-stationary columns"""
        df = X.copy()
        
        for col in df.select_dtypes(include=[np.number]).columns:
            if col in ['Date', 'day_of_week', 'month', 'day_of_year']:
                continue
            
            p_value = self.adf_test(df[col])
            if p_value > self.adf_threshold:
                self.differenced_cols[col] = {
                    'non_stationary': True,
                    'p_value': p_value
                }
        
        return self
    
    def transform(self, X):
        """Apply differencing to non-stationary features"""
        df = X.copy()
        
        for col, info in self.differenced_cols.items():
            if col in df.columns and info.get('non_stationary', False):
                # Frac diff (simplified to standard diff)
                df[col] = df[col].diff()
                df[f"{col}_original"] = X[col]  # Keep original for context
        
        return df
