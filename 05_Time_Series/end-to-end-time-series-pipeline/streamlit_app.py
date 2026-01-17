import sys
import os
from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st

# scripts klasörünü import path'e ekle
sys.path.append(str(Path(__file__).resolve().parents[0] / 'scripts'))
import ts_pipeline as tp

st.set_page_config(page_title='Zaman Serisi Pipeline', layout='wide')

st.title('Zaman Serisi Tahmin - Hızlı Arayüz')
st.markdown('CSV dosyanızı yükleyin (sütunlar: `date`, `target`) ve pipeline otomatik çalışsın.')

uploaded_file = st.file_uploader('CSV dosyasını seçin', type=['csv'])

if uploaded_file is not None:
    # Geçici olarak dosyayı kaydet
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = os.path.join(out_dir, 'uploaded.csv')

    with open(tmp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.success(f'Dosya yüklendi: {tmp_path}')

    # Varsayılan sütun isimleri
    date_col = st.text_input('Date sütunu adı', value='date')
    target_col = st.text_input('Target sütunu adı', value='target')
    freq = st.text_input('Frekans (opsiyonel - örn: D, M)', value='')
    horizon = st.number_input('Tahmin horizon (gün)', min_value=1, max_value=365, value=30)
    n_splits = st.number_input('TimeSeriesSplit fold sayısı', min_value=2, max_value=10, value=5)

    if st.button('Pipeline Çalıştır'):
        with st.spinner('Pipeline çalışıyor - lütfen bekleyin...'):
            try:
                # 1) Load & preprocess
                df = tp.load_and_preprocess(tmp_path, date_col=date_col, target_col=target_col, freq=freq if freq != '' else None, fill_method='interpolate')
                st.write('Veri ön işleme tamamlandı. Örnek:')
                st.dataframe(df.head())

                # 2) Feature engineering
                # load_and_preprocess fonksiyonu hedef sütununu 'target' olarak standartlaştırır
                df_feat = tp.create_features(df, target_col='target')
                model_df = df_feat.dropna().copy()

                # 3) ADF & decomposition
                adf_res = tp.adf_test(df['target'])
                decomp = tp.decompose_series(df['target'], period=7)

                st.write('ADF Testi sonuçları:')
                st.json(adf_res)

                # 4) CV ile eğitim
                metrics, val_preds, rf_models, xgb_models = tp.time_series_cv(model_df, target_col='target', n_splits=int(n_splits))

                # 5) Final fit & forecast
                in_sample_pred, future_df, weights = tp.fit_final_models_and_forecast(model_df, val_preds, rf_models, xgb_models, target_col='target', horizon=int(horizon), output_dir=out_dir)

                # 6) Metrikler
                y_true = model_df['target']
                common_idx = y_true.index.intersection(val_preds.dropna().index)
                y_true_common = y_true.loc[common_idx]
                y_pred_common = val_preds.loc[common_idx]

                from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
                mae = mean_absolute_error(y_true_common, y_pred_common)
                rmse = mean_squared_error(y_true_common, y_pred_common, squared=False)
                r2 = r2_score(y_true_common, y_pred_common)

                st.success('Model eğitimi tamamlandı.')
                st.metric('CV MAE', f"{mae:.4f}")
                st.metric('CV RMSE', f"{rmse:.4f}")
                st.metric('CV R2', f"{r2:.4f}")

                # 7) Grafikler
                img1, img2 = tp.plot_results(df, val_preds, in_sample_pred, future_df, output_dir=out_dir)
                st.image(img1, caption='Gerçek vs Tahmin', use_column_width=True)
                st.image(img2, caption='Gelecek Tahmini', use_column_width=True)

                # 8) PDF raporu oluştur
                executive_summary_placeholder = (
                    "Yönetici Özeti: Buraya kısa, iki-üç paragraf halinde sonuçların ve önerilerin özeti yazılacaktır. "
                    "(Lütfen burayı hedef kitlenize uygun şekilde düzenleyin.)"
                )
                pdf_path = os.path.join(out_dir, 'report.pdf')
                tp.generate_pdf_report(pdf_path, adf_res, decomp, metrics, (img1, img2), executive_summary_placeholder)

                # 9) İndirilebilir dosyalar
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                st.download_button('Raporu İndir (PDF)', data=pdf_bytes, file_name='report.pdf', mime='application/pdf')

                # Modelleri indirme
                with open(os.path.join(out_dir, 'rf_final.joblib'), 'rb') as f:
                    rf_bytes = f.read()
                st.download_button('RandomForest Modeli (joblib)', data=rf_bytes, file_name='rf_final.joblib')

                with open(os.path.join(out_dir, 'xgb_final.joblib'), 'rb') as f:
                    xgb_bytes = f.read()
                st.download_button('XGBoost Modeli (joblib)', data=xgb_bytes, file_name='xgb_final.joblib')

                st.success('Çıktılar hazır ve indirilebilir durumda. `outputs` klasörünü kontrol edin.')

            except Exception as e:
                st.error(f'Hata oluştu: {e}')
                raise

    st.info('Ayarları değiştirdikten sonra "Pipeline Çalıştır" butonuna basın.')
else:
    st.info('CSV dosyasını yükleyin veya sürükleyip bırakın.')
