import sys
from pathlib import Path
import pandas as pd
import numpy as np

# testlerin çalışması için scripts klasörünü path'e ekle
sys.path.append(str(Path(__file__).resolve().parents[1] / 'scripts'))
import ts_pipeline as tp


def test_load_and_preprocess_interpolate():
    # Küçük bir zaman serisi örneği
    df = pd.DataFrame({'date': pd.date_range('2021-01-01', periods=5, freq='D'), 'target': [1, np.nan, 3, np.nan, 5]})
    tmp = 'tmp_test.csv'
    df.to_csv(tmp, index=False)

    res = tp.load_and_preprocess(tmp, date_col='date', target_col='target', freq='D', fill_method='interpolate')
    assert res['target'].isnull().sum() == 0


def test_create_features_lags():
    df = pd.DataFrame({'value': range(1, 11)}, index=pd.date_range('2021-01-01', periods=10, freq='D'))
    df = df.rename(columns={'value': 'target'})
    feat = tp.create_features(df, target_col='target')
    # lag_1'in ikinci satırda first value olması beklenir
    assert feat['lag_1'].iloc[1] == 1
    assert 'lag_7' in feat.columns


def test_metrics():
    y_true = [1,2,3,4]
    y_pred = [1.1,1.9,3.2,3.8]
    mae = np.mean(np.abs(np.array(y_true)-np.array(y_pred)))
    rmse = np.sqrt(np.mean((np.array(y_true)-np.array(y_pred))**2))
    # tp modulündeki sklearn metricleri ile karşılaştır
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    assert abs(mae - mean_absolute_error(y_true, y_pred)) < 1e-8
    assert abs(rmse - (mean_squared_error(y_true, y_pred) ** 0.5)) < 1e-8
