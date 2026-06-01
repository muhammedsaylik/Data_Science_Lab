import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings('ignore')


def calculate_vif(df, numeric_cols=None):
    """Variance Inflation Factor hesaplar"""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df[numeric_cols].dropna()
    
    if df_clean.empty or df_clean.shape[1] < 2:
        return pd.DataFrame({'Feature': [], 'VIF': []})
    
    vif_data = pd.DataFrame()
    vif_data['Feature'] = df_clean.columns
    
    try:
        vif_data['VIF'] = [variance_inflation_factor(df_clean.values, i) 
                           for i in range(df_clean.shape[1])]
    except:
        vif_data['VIF'] = np.nan
    
    return vif_data.sort_values('VIF', ascending=False)


def test_stationarity(series, name='Series', alpha=0.05):
    """Augmented Dickey-Fuller ve KPSS testleri yapar"""
    results = {}
    
    # ADF test
    adf_result = adfuller(series.dropna(), autolag='AIC')
    results['ADF_Statistic'] = adf_result[0]
    results['ADF_P_Value'] = adf_result[1]
    results['ADF_Critical_5%'] = adf_result[4]['5%']
    results['ADF_Stationary'] = 'Yes' if adf_result[1] < alpha else 'No'
    
    # KPSS test
    try:
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')
        results['KPSS_Statistic'] = kpss_result[0]
        results['KPSS_P_Value'] = kpss_result[1]
        results['KPSS_Critical_5%'] = kpss_result[3]['5%']
        results['KPSS_Stationary'] = 'No' if kpss_result[1] < alpha else 'Yes'
    except:
        results['KPSS_Stationary'] = 'Error'
    
    return results


def analyze_stationarity_for_features(df, numeric_cols=None):
    """Tüm özellikler için durağanlık testleri yapar"""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stationarity_results = []
    
    for col in numeric_cols:
        valid_data = df[col].dropna()
        if len(valid_data) > 20 and valid_data.std() > 1e-10:
            try:
                results = test_stationarity(df[col], name=col)
                results['Feature'] = col
                stationarity_results.append(results)
            except:
                pass
    
    return pd.DataFrame(stationarity_results) if stationarity_results else pd.DataFrame()


def calculate_correlation_matrix(df, numeric_cols=None, method='pearson'):
    """Korelasyon matrisi hesaplar"""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    return df[numeric_cols].corr(method=method)


def identify_highly_correlated_features(corr_matrix, threshold=0.95):
    """Yüksek korelasyonlu özellik çiftlerini bulur"""
    high_corr_pairs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    return pd.DataFrame(high_corr_pairs)


def select_features_by_variance(df, numeric_cols=None, threshold=0.01):
    """Düşük varyansı olan özellikleri filtreler"""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    variances = df[numeric_cols].var()
    
    low_var_features = variances[variances < threshold]
    
    return {
        'low_variance_features': low_var_features.index.tolist(),
        'high_variance_features': variances[variances >= threshold].index.tolist(),
        'variance_stats': variances
    }


def analyze_feature_statistics(df, numeric_cols=None):
    """Özelliklerin temel istatistiklerini hesaplar"""
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats_data = []
    
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) > 0:
            stats_data.append({
                'Feature': col,
                'Count': len(data),
                'Mean': data.mean(),
                'Std': data.std(),
                'Min': data.min(),
                'Max': data.max(),
                'Skewness': stats.skew(data),
                'Kurtosis': stats.kurtosis(data),
                'Missing_Count': df[col].isna().sum()
            })
    
    return pd.DataFrame(stats_data)


def select_optimal_features(df, target_col=None, numeric_cols=None, 
                            vif_threshold=10, corr_threshold=0.95,
                            var_threshold=0.01, protected_features=None):
    """
    Optimal özellikler seçer
    
    Parameters:
    -----------
    protected_features : list
        VIF ve filtrelerden korunacak kritik özellikler (Sentiment/News features)
        Otomatik silinmesinden korunurlar ve Model Priority olarak işaretlenirler
    """
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if protected_features is None:
        protected_features = []
    
    protected_set = set(protected_features)
    selected_features = set(numeric_cols)
    removed_features = {}
    protected_info = {
        'protected_features': list(protected_set),
        'description': 'Model Priority - Protected from automatic filtering'
    }
    
    # 1. Düşük varyans özellikleri kaldır (protected exclude)
    var_result = select_features_by_variance(df, numeric_cols, threshold=var_threshold)
    low_var = set(var_result['low_variance_features']) - protected_set
    selected_features -= low_var
    removed_features['low_variance'] = list(low_var)
    
    # 2. Eksik veri oranı yüksek özellikleri kaldır (protected exclude)
    missing_ratio = df[numeric_cols].isna().sum() / len(df)
    high_missing = set(missing_ratio[missing_ratio > 0.5].index) - protected_set
    selected_features -= high_missing
    removed_features['high_missing'] = list(high_missing)
    
    # 3. VIF yüksek özellikleri kaldır (protected exclude)
    numeric_from_selected = list(selected_features)
    if len(numeric_from_selected) > 1:
        vif_data = calculate_vif(df[numeric_from_selected])
        high_vif_all = set(vif_data[vif_data['VIF'] > vif_threshold]['Feature'].values)
        high_vif = high_vif_all - protected_set
        
        if len(high_vif) > 0:
            selected_features -= high_vif
            removed_features['high_vif'] = list(high_vif)
        
        protected_high_vif = high_vif_all & protected_set
        if len(protected_high_vif) > 0:
            protected_info['high_vif_protection'] = list(protected_high_vif)
    
    # 4. Yüksek korelasyonlu özellikler arasından seç (protected priority)
    numeric_from_selected = list(selected_features)
    if len(numeric_from_selected) > 1:
        corr_matrix = df[numeric_from_selected].corr()
        high_corr = identify_highly_correlated_features(corr_matrix, threshold=corr_threshold)
        
        if len(high_corr) > 0:
            to_remove = []
            for _, row in high_corr.iterrows():
                f1, f2 = row['Feature_1'], row['Feature_2']
                
                if f1 in protected_set and f2 not in protected_set:
                    to_remove.append(f2)
                elif f2 in protected_set and f1 not in protected_set:
                    to_remove.append(f1)
                elif f1 not in protected_set and f2 not in protected_set:
                    var1 = df[f1].var()
                    var2 = df[f2].var()
                    to_remove_col = f1 if var1 < var2 else f2
                    if to_remove_col in selected_features:
                        to_remove.append(to_remove_col)
            
            selected_features -= set(to_remove)
            removed_features['high_correlation'] = list(set(to_remove))
    
    return {
        'selected_features': sorted(list(selected_features)),
        'removed_features': removed_features,
        'protected_features': protected_info,
        'total_features_removed': len(numeric_cols) - len(selected_features)
    }


def check_data_leakage_features(feature_names):
    """Potansiyel data leakage'ı kontrol eder"""
    leakage_patterns = {
        'future': ['lead', 'future', 'tomorrow', 'next_'],
        'target_related': ['target', 'label', 'y_'],
        'data_artifacts': ['date_id', 'time_id']
    }
    
    suspicious_features = []
    
    for feature in feature_names:
        for category, patterns in leakage_patterns.items():
            for pattern in patterns:
                if pattern.lower() in feature.lower():
                    suspicious_features.append({
                        'Feature': feature,
                        'Category': category,
                        'Pattern': pattern
                    })
    
    return pd.DataFrame(suspicious_features) if suspicious_features else None


def calculate_target_correlation(df, target_col, numeric_cols=None, lag=0):
    """
    Calculate correlation between features and target
    Critical: Identifies which features actually predict the target
    
    Parameters
    ----------
    target_col : str
        Target variable name
    lag : int
        Shift target backwards by lag periods (prevents lookahead bias)
    """
    if target_col not in df.columns:
        return pd.DataFrame()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    target_shifted = df[target_col].shift(-lag)  # Shift target to prevent future leakage
    
    correlations = []
    for col in numeric_cols:
        if col == target_col:
            continue
        
        valid_mask = df[[col, target_col]].notna().all(axis=1)
        if valid_mask.sum() < 10:
            continue
        
        corr = df.loc[valid_mask, col].corr(target_shifted[valid_mask])
        correlations.append({
            'Feature': col,
            'Target_Correlation': corr,
            'Abs_Correlation': abs(corr)
        })
    
    result = pd.DataFrame(correlations).sort_values('Abs_Correlation', ascending=False)
    return result


def detect_lookahead_bias(df, feature_cols, target_col, window=20):
    """
    Detect potential lookahead bias in features
    Check if rolling window features use future data
    
    Parameters
    ----------
    feature_cols : list
        Features to check
    target_col : str
        Target variable
    window : int
        Rolling window size to check
    
    Returns
    -------
    bias_report : dict
        Suspicious features and their correlation with future target
    """
    
    bias_report = {}
    
    if target_col not in df.columns:
        return bias_report
    
    # Check correlation with FUTURE target (1, 5, 10 steps ahead)
    for look_ahead_steps in [1, 5, 10]:
        future_target = df[target_col].shift(-look_ahead_steps)
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            valid_mask = df[[col, target_col]].notna().all(axis=1)
            if valid_mask.sum() < 10:
                continue
            
            # Correlation with FUTURE target
            future_corr = df.loc[valid_mask, col].corr(future_target[valid_mask])
            
            # If feature has HIGH correlation with future target -> LEAKAGE
            if abs(future_corr) > 0.3:
                key = f"{col}_lookahead_{look_ahead_steps}d"
                bias_report[key] = {
                    'Feature': col,
                    'Look_Ahead_Steps': look_ahead_steps,
                    'Future_Target_Correlation': future_corr,
                    'Risk_Level': 'HIGH' if abs(future_corr) > 0.5 else 'MEDIUM'
                }
    
    return bias_report


def validate_stationarity_for_modeling(df, numeric_cols=None, enforce_differencing=True):
    """
    Validate stationarity and suggest differencing if needed
    Non-stationary features WILL CAUSE negative R² in linear models
    
    Parameters
    ----------
    enforce_differencing : bool
        If True, automatically difference all non-stationary features
    
    Returns
    -------
    validation: dict
        Stationarity status and transformations
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    validation = {
        'stationary': [],
        'non_stationary': [],
        'transformations': {}
    }
    
    for col in numeric_cols:
        valid_data = df[col].dropna()
        if len(valid_data) > 20 and valid_data.std() > 1e-10:
            try:
                adf_result = adfuller(valid_data, autolag='AIC')
                p_value = adf_result[1]
                
                if p_value < 0.05:
                    validation['stationary'].append(col)
                else:
                    validation['non_stationary'].append(col)
                    
                    if enforce_differencing:
                        # Record differencing transformation
                        validation['transformations'][col] = {
                            'method': 'differencing',
                            'reason': f'Non-stationary (ADF p={p_value:.4f})',
                            'differencing_order': 1
                        }
            except:
                pass
    
    return validation


def create_feature_selection_report(df, target_col=None, numeric_cols=None):
    """Kapsamlı feature selection raporu oluşturur"""
    report = {}
    
    # İstatistikler
    report['statistics'] = analyze_feature_statistics(df, numeric_cols)
    
    # Durağanlık
    report['stationarity'] = analyze_stationarity_for_features(df, numeric_cols)
    stationarity_validation = validate_stationarity_for_modeling(df, numeric_cols)
    report['stationarity_validation'] = stationarity_validation
    
    # VIF
    numeric_cols_for_vif = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols_for_vif) > 1:
        report['vif'] = calculate_vif(df, numeric_cols_for_vif)
    
    # Korelasyon
    corr_matrix = calculate_correlation_matrix(df, numeric_cols)
    report['correlation_matrix'] = corr_matrix
    report['high_correlation'] = identify_highly_correlated_features(corr_matrix, threshold=0.95)
    
    # Target Correlation (Critical!)
    if target_col and target_col in df.columns:
        report['target_correlation'] = calculate_target_correlation(df, target_col, numeric_cols)
        
        # Lookahead Bias Detection
        numeric_cols_list = numeric_cols if numeric_cols else df.select_dtypes(include=[np.number]).columns.tolist()
        lookahead_bias = detect_lookahead_bias(df, numeric_cols_list, target_col)
        if lookahead_bias:
            report['lookahead_bias_detected'] = lookahead_bias
    
    # Optimal seçim
    selection = select_optimal_features(df, target_col, numeric_cols)
    report['feature_selection'] = selection
    
    # Data leakage
    leakage = check_data_leakage_features(df.columns.tolist())
    if leakage is not None:
        report['potential_leakage'] = leakage
    
    return report
