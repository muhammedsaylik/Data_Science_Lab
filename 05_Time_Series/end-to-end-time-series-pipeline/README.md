# Zaman Serisi Tahmin Pipeline

Bu proje, kullanıcının yükleyeceği bir CSV (sütunlar: `date`, `target`) üzerinden uçtan uca bir zaman serisi işleme ve tahmin pipeline'ı sağlar.

Özellikler:
- Eksik değer doldurma (interpolate / forward-fill)
- Aykırı değer tespiti ve caps'leme (IQR yöntemi)
- Özellik mühendisliği (gün, ay, yıl, haftanın günü, hafta sonu, lag_1, lag_7)
- ADF testi ve seasonal decomposition
- Modeller: RandomForest ve XGBoost ile TimeSeriesSplit CV
- Basit weighted ensemble ve 30 günlük rekürsif tahmin
- Görselleştirme ve PDF raporu (FPDF)

Kurulum:
1. Sanal ortam oluşturun (opsiyonel)
2. `pip install -r requirements.txt`

Çalıştırma örneği:
```
python scripts/ts_pipeline.py --data path/to/your.csv --date_col date --target_col target --output_dir outputs --horizon 30
```

Notlar:
- CSV dosyanız `date` adıyla tarih, `target` adıyla hedef sütununu içermelidir. İsterseniz farklı sütun isimleri için argümanlarla belirtebilirsiniz.
- `outputs` klasöründe: model dosyaları, görseller ve `report.pdf` raporu oluşacaktır.

Yönetici Özeti:
- `scripts/ts_pipeline.py` içinde yer alan `executive_summary_placeholder` değişkenini, raporda kullanılacak şekilde düzenleyebilirsiniz.
