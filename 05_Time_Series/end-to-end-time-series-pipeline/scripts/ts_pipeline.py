#!/usr/bin/env python3
"""
Zaman Serisi Pipeline Scripti
- Veri yükleme ve ön işleme (eksik değer doldurma, aykırı değerleri IQR ile işaretleme/düzeltme)
- Özellik mühendisliği (tarihten özellikler, lagler)
- İstatistiksel testler (ADF, seasonal_decompose)
- Modeller: RandomForest ve XGBoost, TimeSeriesSplit ile CV
- Basit bir Weighted Ensemble ve gelecek 30 gün için rekürsif tahmin
- Metrikler: MAE, RMSE, R2
- Görselleştirme ve PDF raporu üretimi

Nasıl çalıştırılır:
python scripts/ts_pipeline.py --data PATH_TO_CSV --date_col date --target_col target --output_dir outputs --horizon 30

"""

import argparse
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Temel kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from fpdf import FPDF
import joblib
import urllib.request
import unicodedata

# ----------------------------- Yardımcı Fonksiyonlar -----------------------------

def load_and_preprocess(csv_path, date_col='date', target_col='target', freq=None, fill_method='interpolate'):
    """CSV'i okur, tarih parse eder, eksik tarihleri tamamlar ve eksik değerleri doldurur."""
    # Veri yükle
    df = pd.read_csv(csv_path)
    # Tarih sütunu datetime'a çevir ve sıralama
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    # Ensure provided target column exists (case-insensitive match)
    if target_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == target_col.lower()]
        if matches:
            target_col = matches[0]
        else:
            raise KeyError(f"Target column '{target_col}' not found in CSV. Available columns: {list(df.columns)}")

    df = df[[date_col, target_col]].copy()
    # Standardize target column name to 'target' for downstream consistency
    df = df.rename(columns={target_col: 'target'})

    # Index olarak tarih
    df.set_index(date_col, inplace=True)

    # Frekansı tahmin etme (günlük varsayılan)
    inferred = pd.infer_freq(df.index)
    if freq is None:
        freq = inferred if inferred is not None else 'D'
    print(f"Tahmini frekans: {inferred} -> kullanılıyor: {freq}")

    # Tüm tarih aralığını doldur
    full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df.reindex(full_idx)

    # Eksik değerleri doldurma
    if fill_method == 'interpolate':
        # Zaman serisine uygun interpolasyon
        df['target'] = df['target'].interpolate(method='time')
        # fillna(method=...) deprecated; use bfill/ffill
        df['target'] = df['target'].bfill().ffill()
    else:
        # forward/backward fill using recommended methods
        df['target'] = df['target'].ffill().bfill()

    # Aykırı değer tespiti ve düzeltme (IQR)
    Q1 = df['target'].quantile(0.25)
    Q3 = df['target'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df['is_outlier'] = ((df['target'] < lower) | (df['target'] > upper)).astype(int)
    # Aykırı değerleri caps'leme (winsorize)
    df['target'] = df['target'].clip(lower=lower, upper=upper)

    return df


def create_features(df, target_col='target'):
    """Tarih özellikleri ve lag üretimi (t-1, t-7)
    df: index'i datetime olan DataFrame
    """
    df = df.copy()
    # Tarihten özellikler çıkar
    df['day'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['weekday'] = df.index.weekday
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)

    # Lag özellikleri
    df['lag_1'] = df[target_col].shift(1)
    df['lag_7'] = df[target_col].shift(7)

    return df


def adf_test(series):
    """ADF testi uygular ve sonuçları döndürür. Hata durumunda uyarı bilgilerinin yer aldığı bir sözlük döndürür."""
    try:
        res = adfuller(series.dropna())
        adf_stat = res[0]
        p_value = res[1]
        usedlag = res[2]
        nobs = res[3]
        crit = res[4]
        return {
            'adf_stat': adf_stat,
            'p_value': p_value,
            'usedlag': usedlag,
            'nobs': nobs,
            'crit_vals': crit
        }
    except Exception as e:
        # Veri çok kısa veya başka bir hata oluştu
        return {
            'adf_stat': float('nan'),
            'p_value': float('nan'),
            'usedlag': None,
            'nobs': None,
            'crit_vals': {},
            'error': str(e)
        }


def decompose_series(series, period=None, model='additive'):
    """Seasonal decomposition uygular. Period belirtilmezse haftalık 7 tercih edilir. Hata durumunda None döndürür."""
    if period is None:
        period = 7  # günlük veri için haftalık mevsimsellik
    try:
        decomposition = seasonal_decompose(series.dropna(), model=model, period=period)
        return decomposition
    except Exception as e:
        # Yetersiz veri veya başka bir hata olabilir
        return None


def time_series_cv(df, target_col='target', n_splits=5, random_state=42):
    """TimeSeriesSplit ile modelleri eğitir ve doğrulama metriklerini döndürür."""
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    # Özellik olarak numeric olanları seç
    X = X.select_dtypes(include=[np.number]).fillna(0)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = []
    val_preds = pd.Series(index=y.index, dtype=float)

    rf_val_preds = []
    xgb_val_preds = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"Fold {fold + 1}/{n_splits} - train:{train_idx[0]}..{train_idx[-1]} val:{val_idx[0]}..{val_idx[-1]}")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Modeller
        rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
        xgb = XGBRegressor(n_estimators=200, random_state=random_state, verbosity=0)

        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)

        y_pred_rf = rf.predict(X_val)
        y_pred_xgb = xgb.predict(X_val)

        # Ensemble ağırlıkları basitçe 1/RMSE kullanılarak hesaplanır
        rmse_rf = mean_squared_error(y_val, y_pred_rf) ** 0.5
        rmse_xgb = mean_squared_error(y_val, y_pred_xgb) ** 0.5
        w_rf = (1 / rmse_rf) if rmse_rf > 0 else 0.5
        w_xgb = (1 / rmse_xgb) if rmse_xgb > 0 else 0.5
        total = w_rf + w_xgb
        w_rf /= total
        w_xgb /= total

        y_pred_ens = w_rf * y_pred_rf + w_xgb * y_pred_xgb

        ens_rmse = mean_squared_error(y_val, y_pred_ens) ** 0.5

        fold_metrics = {
            'fold': fold + 1,
            'rf_rmse': rmse_rf,
            'xgb_rmse': rmse_xgb,
            'ens_rmse': ens_rmse,
            'rf_mae': mean_absolute_error(y_val, y_pred_rf),
            'xgb_mae': mean_absolute_error(y_val, y_pred_xgb),
            'ens_mae': mean_absolute_error(y_val, y_pred_ens),
            'rf_r2': r2_score(y_val, y_pred_rf),
            'xgb_r2': r2_score(y_val, y_pred_xgb),
            'ens_r2': r2_score(y_val, y_pred_ens),
            'w_rf': w_rf,
            'w_xgb': w_xgb
        }
        metrics.append(fold_metrics)

        # Sakla
        val_preds.iloc[val_idx] = y_pred_ens

        # Kaydet (daha sonra ensemble ağırlıklarını ortalamak için)
        rf_val_preds.append((rf, w_rf))
        xgb_val_preds.append((xgb, w_xgb))

    return metrics, val_preds, rf_val_preds, xgb_val_preds


def fit_final_models_and_forecast(df, val_preds, rf_models, xgb_models, target_col='target', horizon=30, output_dir='outputs'):
    """Son modelleri tüm veriye fit eder ve gelecek horizon kadar rekürsif tahmin yapar."""
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col]

    # Son modellerin ağırlıklarını ortala
    w_rf = np.mean([w for (_, w) in rf_models])
    w_xgb = np.mean([w for (_, w) in xgb_models])
    total = w_rf + w_xgb
    w_rf /= total
    w_xgb /= total
    print(f"Ensemble ağırlıkları - RF: {w_rf:.3f}, XGB: {w_xgb:.3f}")

    # Final modelleri tüm veriye fit et
    rf_final = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    xgb_final = XGBRegressor(n_estimators=300, random_state=42, verbosity=0)
    rf_final.fit(X, y)
    xgb_final.fit(X, y)

    # Gelecek için iterative forecast
    last_date = df.index.max()
    freq = df.index.inferred_freq if df.index.inferred_freq is not None else 'D'

    future_index = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq), periods=horizon, freq=freq)

    # Başlangıç olarak mevcut seriyi bir listeye alıp yeni tahminleri ekleyeceğiz
    series = df[target_col].tolist()
    future_preds = []

    # Eğitimde kullanılan özellik sütunlarını kaydet
    feature_cols = X.columns.tolist()
    last_row = df.iloc[-1].to_dict()

    for d in future_index:
        # Her bir feature için uygun değeri üret (tarih-tabanlı ve lag'ler)
        row = {}
        for col in feature_cols:
            if col == 'day':
                row[col] = d.day
            elif col == 'month':
                row[col] = d.month
            elif col == 'year':
                row[col] = d.year
            elif col == 'weekday':
                row[col] = d.weekday()
            elif col == 'is_weekend':
                row[col] = int(d.weekday() in [5,6])
            elif col.startswith('lag_'):
                # lag_N şeklinde varsayılıyor
                try:
                    n = int(col.split('_')[1])
                except Exception:
                    n = 1
                if len(series) >= n:
                    row[col] = series[-n]
                else:
                    row[col] = series[0]
            else:
                # Diğer sayısal özellikler için son bilinen değeri veya 0
                row[col] = last_row.get(col, 0)

        # DataFrame şeklinde ve aynı sütun sırasıyla
        X_row_df = pd.DataFrame([row], index=[d])[feature_cols]

        # Modellerin tahmini
        pred_rf = rf_final.predict(X_row_df)[0]
        pred_xgb = xgb_final.predict(X_row_df)[0]
        pred = w_rf * pred_rf + w_xgb * pred_xgb

        future_preds.append(pred)
        # Rekürsif olarak ekle
        series.append(pred)

    future_df = pd.DataFrame({ 'ds': future_index, 'yhat': future_preds }).set_index('ds')

    # Train üzerindeki son dönemin tahminini de hesaplayalım (fit edilen modellerle)
    in_sample_pred = w_rf * rf_final.predict(X) + w_xgb * xgb_final.predict(X)
    in_sample_pred = pd.Series(in_sample_pred, index=df.index)

    # Model nesnelerini kaydet
    joblib.dump(rf_final, os.path.join(output_dir, 'rf_final.joblib'))
    joblib.dump(xgb_final, os.path.join(output_dir, 'xgb_final.joblib'))

    return in_sample_pred, future_df, {'w_rf': w_rf, 'w_xgb': w_xgb}


def plot_results(df, val_preds, in_sample_pred, future_df, output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)

    sns.set(style='whitegrid', context='talk')

    # Gerçek vs Tahmin (Validation / In-sample)
    plt.figure(figsize=(14,6))
    # Hedef sütunu yoksa, ilk sayısal sütunu kullanmayı dene
    if 'target' in df.columns:
        y_series = df['target']
    else:
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        y_series = df[nums[0]] if nums else pd.Series(index=df.index, data=np.nan)

    plt.plot(df.index, y_series, label='Gerçek', color='black')
    plt.plot(val_preds.index, val_preds.values, label='CV Tahmini (Ensemble)', color='orange', alpha=0.8)
    plt.plot(in_sample_pred.index, in_sample_pred.values, label='Final Model (in-sample)', color='green', alpha=0.6)
    plt.legend()
    plt.title('Gerçek vs Tahmin')
    plt.xlabel('Tarih')
    plt.ylabel('Target')
    plt.tight_layout()
    path1 = os.path.join(output_dir, 'actual_vs_pred.png')
    plt.savefig(path1)
    plt.close()

    # Gelecek tahmini grafiği
    plt.figure(figsize=(14,6))
    plt.plot(df.index, df['target'], label='Gerçek (Geçmiş)', color='black')
    plt.plot(future_df.index, future_df['yhat'], label='Gelecek Tahmini', color='red', linestyle='--')
    plt.legend()
    plt.title('Gelecek Tahmini')
    plt.xlabel('Tarih')
    plt.ylabel('Target')
    plt.tight_layout()
    path2 = os.path.join(output_dir, 'future_forecast.png')
    plt.savefig(path2)
    plt.close()

    return path1, path2


def generate_pdf_report(output_path, adf_res, decomp, metrics, images, executive_summary_placeholder):
    """Basit bir PDF raporu oluşturur (fpdf kullanılarak). UTF-8 karakterleri desteklemek için DejaVuSans.ttf indirip kullanmaya çalışır; başarısız olursa ASCII sanitizasyonu uygulanır."""
    # Fonts klasörü oluştur ve DejaVuSans.ttf indirilmeye çalışılsın
    fonts_dir = os.path.join(os.path.dirname(output_path), 'fonts')
    os.makedirs(fonts_dir, exist_ok=True)
    font_path = os.path.join(fonts_dir, 'DejaVuSans.ttf')
    if not os.path.exists(font_path):
        try:
            url = 'https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf'
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            # İndirme başarısızsa devam edeceğiz; ASCII sanitizasyonu uygulanacak
            font_path = None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Eğer font mevcutsa eklemeye çalış
    font_added = False
    if font_path and os.path.exists(font_path):
        try:
            pdf.add_page()
            pdf.add_font('DejaVu', '', font_path, uni=True)
            pdf.set_font('DejaVu', 'B', 16)
            font_added = True
        except Exception:
            font_added = False

    # Basit helper: eğer Unicode desteklenmiyorsa metinleri ASCII'ye indirger
    def _safe_text(s):
        if font_added:
            return str(s)
        s = str(s)
        return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

    # Başlık sayfası
    if not font_added:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, _safe_text('Zaman Serisi Analiz Raporu'), ln=True, align='C')
    pdf.ln(6)

    pdf.set_font('Arial', size=11)
    # ADF sonuçlarını güvenli şekilde yaz
    try:
        pdf.multi_cell(0, 6, f"ADF Testi Sonuçları: ADF istatistiği = {adf_res.get('adf_stat', float('nan')):.4f}, p-değeri = {adf_res.get('p_value', float('nan')):.4f}")
    except Exception:
        pdf.multi_cell(0, 6, f"ADF Testi Sonuçları: bilgi alınamadı. Raw: {adf_res}")
    pdf.ln(4)
    if decomp is not None:
        pdf.multi_cell(0, 6, "Mevsimsellik (Seasonal Decomposition): Observed, Trend, Seasonal ve Residual bileşenleri üretildi.")
    else:
        pdf.multi_cell(0, 6, "Mevsimsellik analizi yapılamadı veya yetersiz veri nedeniyle atlandı.")
    pdf.ln(6)

    # Metikler
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Çapraz Doğrulama Metriği Özetleri', ln=True)
    pdf.set_font('Arial', size=10)
    for m in metrics:
        s = f"Fold {m['fold']}: ENS_RMSE={m['ens_rmse']:.4f}, ENS_MAE={m['ens_mae']:.4f}, ENS_R2={m['ens_r2']:.4f}"
        pdf.multi_cell(0, 6, s)
    pdf.ln(6)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Grafikler', ln=True)
    pdf.image(images[0], w=180)
    pdf.ln(6)
    pdf.image(images[1], w=180)
    pdf.ln(8)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, 'Yönetici Özeti (Placeholder)', ln=True)
    pdf.set_font('Arial', size=10)
    pdf.multi_cell(0, 6, executive_summary_placeholder)

    pdf.output(output_path)
    print(f"PDF raporu kaydedildi: {output_path}")


# ----------------------------- Ana İş Akışı -----------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) Veri yükle & ön işle
    print("1) Veri yükleniyor ve ön işleme uygulanıyor...")
    df = load_and_preprocess(args.data, date_col=args.date_col, target_col=args.target_col, freq=args.freq, fill_method='interpolate')

    # 2) Özellik mühendisliği
    print("2) Özellikler üretiliyor...")
    # create_features fonksiyonu pipeline genelinde 'target' sütununu varsayar
    df_feat = create_features(df, target_col='target')

    # Hedef ve özellikler
    model_df = df_feat.dropna().copy()  # lag sebebiyle oluşan NA'ları düşür

    # 3) İstatistiksel testler
    print("3) ADF testi ve mevsimsellik analizleri uygulanıyor...")
    adf_res = adf_test(df['target'])
    # decomposition (günlük veri için varsayılan periyot 7)
    decomp = decompose_series(df['target'], period=7)

    # 4) Model eğitimi (TimeSeriesSplit)
    print("4) Modeller TimeSeriesSplit ile CV yapıyor...")
    metrics, val_preds, rf_models, xgb_models = time_series_cv(model_df, target_col='target', n_splits=args.n_splits)

    # CV sonuçlarının özetlenmesi
    print("CV sonuçları (orta değerler):")
    import statistics
    ens_rmse_median = statistics.median([m['ens_rmse'] for m in metrics])
    ens_mae_median = statistics.median([m['ens_mae'] for m in metrics])
    print(f"Ensemble median RMSE: {ens_rmse_median:.4f}, MAE: {ens_mae_median:.4f}")

    # 5) Final fit ve gelecek tahmini
    print("5) Final modeller fit ediliyor ve gelecek tahmini yapılıyor...")
    in_sample_pred, future_df, weights = fit_final_models_and_forecast(model_df, val_preds, rf_models, xgb_models, target_col='target', horizon=args.horizon, output_dir=args.output_dir)

    # 6) Metrikler (son CV tahmini üzerine)
    print("6) Metrikler hesaplanıyor...")
    # CV üzerinden global metrikler
    y_true = model_df['target']
    # Val_preds içinde bazı indexler NaN olabilir (ilk train bölümleri), bunları düşürelim
    common_idx = y_true.index.intersection(val_preds.dropna().index)
    y_true_common = y_true.loc[common_idx]
    y_pred_common = val_preds.loc[common_idx]

    mae = mean_absolute_error(y_true_common, y_pred_common)
    rmse = mean_squared_error(y_true_common, y_pred_common) ** 0.5
    r2 = r2_score(y_true_common, y_pred_common)
    print(f"CV Genel: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # 7) Görselleştirme
    print("7) Grafikler üretiliyor...")
    path1, path2 = plot_results(df, val_preds, in_sample_pred, future_df, output_dir=args.output_dir)

    # 8) PDF raporu
    print("8) PDF raporu oluşturuluyor...")
    executive_summary_placeholder = (
        "Yönetici Özeti: Buraya kısa, iki-üç paragraf halinde sonuçların ve önerilerin özeti yazılacaktır. "
        "(Lütfen burayı hedef kitlenize uygun şekilde düzenleyin.)"
    )
    pdf_path = os.path.join(args.output_dir, 'report.pdf')
    generate_pdf_report_safe(pdf_path, adf_res, decomp, metrics, (path1, path2), executive_summary_placeholder)

    print("Tamamlandı. Çıktılar ve modeller `" + args.output_dir + "` içinde saklandı.")


def generate_pdf_report_safe(output_path, adf_res, decomp, metrics, images, executive_summary_placeholder):
    """Daha dayanıklı PDF üreticisi. UTF-8 fontu indirmeyi dener; yoksa ASCII sanitizasyonu uygular."""
    fonts_dir = os.path.join(os.path.dirname(output_path), 'fonts')
    os.makedirs(fonts_dir, exist_ok=True)
    font_path = os.path.join(fonts_dir, 'DejaVuSans.ttf')
    if not os.path.exists(font_path):
        try:
            url = 'https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf'
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            font_path = None

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    font_added = False
    if font_path and os.path.exists(font_path):
        try:
            pdf.add_font('DejaVu', '', font_path, uni=True)
            font_added = True
        except Exception:
            font_added = False

    def _safe_text(s):
        if font_added:
            return str(s)
        s = str(s)
        return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

    pdf.add_page()
    if font_added:
        pdf.set_font('DejaVu', 'B', 16)
    else:
        pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, _safe_text('Zaman Serisi Analiz Raporu'), ln=True, align='C')
    pdf.ln(6)

    if font_added:
        pdf.set_font('DejaVu', '', 12)
    else:
        pdf.set_font('Arial', '', 11)

    try:
        pdf.multi_cell(0, 6, _safe_text(f"ADF Testi Sonuçları: ADF istatistiği = {adf_res.get('adf_stat', float('nan')):.4f}, p-değeri = {adf_res.get('p_value', float('nan')):.4f}"))
    except Exception:
        pdf.multi_cell(0, 6, _safe_text(f"ADF Testi Sonuçları: bilgi alinamadi. Raw: {adf_res}"))
    pdf.ln(4)
    if decomp is not None:
        pdf.multi_cell(0, 6, _safe_text("Mevsimsellik (Seasonal Decomposition): Observed, Trend, Seasonal ve Residual bileşenleri uretildi."))
    else:
        pdf.multi_cell(0, 6, _safe_text("Mevsimsellik analizi yapilamadi veya yetersiz veri nedeniyle atlandi."))
    pdf.ln(6)

    if font_added:
        pdf.set_font('DejaVu', 'B', 12)
    else:
        pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, _safe_text('Capraz Dogrulama Metrigi Ozetleri'), ln=True)
    if font_added:
        pdf.set_font('DejaVu', '', 10)
    else:
        pdf.set_font('Arial', '', 10)
    for m in metrics:
        s = f"Fold {m['fold']}: ENS_RMSE={m['ens_rmse']:.4f}, ENS_MAE={m['ens_mae']:.4f}, ENS_R2={m['ens_r2']:.4f}"
        pdf.multi_cell(0, 6, _safe_text(s))
    pdf.ln(6)

    try:
        if images[0] and os.path.exists(images[0]):
            pdf.image(images[0], w=180)
            pdf.ln(6)
        if images[1] and os.path.exists(images[1]):
            pdf.image(images[1], w=180)
            pdf.ln(8)
    except Exception:
        pass

    if font_added:
        pdf.set_font('DejaVu', 'B', 12)
    else:
        pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 8, _safe_text('Yonetici Ozeti (Placeholder)'), ln=True)
    if font_added:
        pdf.set_font('DejaVu', '', 10)
    else:
        pdf.set_font('Arial', '', 10)
    pdf.multi_cell(0, 6, _safe_text(executive_summary_placeholder))

    try:
        pdf.output(output_path)
        print(f"PDF raporu kaydedildi: {output_path}")
    except UnicodeEncodeError:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 6, _safe_text('Rapor olusturulamadi, non-ascii karakterler nedeni ile sinirli rapor uretildi.'))
        pdf.output(output_path)
        print(f"PDF raporu (sanitized) kaydedildi: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zaman Serisi Uçtan Uca Pipeline')
    parser.add_argument('--data', type=str, required=True, help='CSV dosya yolu (date ve target sütunlarını içermeli)')
    parser.add_argument('--date_col', type=str, default='date', help='Tarih sütunu adı')
    parser.add_argument('--target_col', type=str, default='target', help='Tahmin edilecek hedef sütun adı')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Çıktı klasörü')
    parser.add_argument('--horizon', type=int, default=30, help='Gelecek için tahmin horizon (gün)')
    parser.add_argument('--n_splits', type=int, default=5, help='TimeSeriesSplit için fold sayısı')
    parser.add_argument('--freq', type=str, default=None, help='Opsiyonel: veri frekansını belirtin (örn: D, M)')
    args = parser.parse_args()

    main(args)
