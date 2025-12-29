# Hotel Booking Demand: Feature Engineering & EDA

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-v1.x-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Latest-013243?style=flat-square&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-0071ba?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Latest-2ca02c?style=flat-square)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-f7931e?style=flat-square&logo=scikit-learn)

## Proje Özeti

Bu proje, bir otel rezervasyon veri seti üzerinde uçtan uca veri temizleme ve özellik mühendisliği (Feature Engineering) süreçlerine odaklanmaktadır. Ham veriden modelin daha etkili öğrenebileceği anlamlı değişkenler türeterek, misafirlerin iptal davranışları sistematik olarak analiz edilmiştir.

---

## Proje Yapısı

```
C:.
│   README.md
│
├───data/
│   ├───processed/
│   │   └── hotel_bookings_featured.csv
│   │
│   └───raw/
│       └── hotel_bookings.csv
│
├───notebooks/
│   └── Hotel_booking_demand.ipynb
│
|───reports/
```

---

## Metodoloji

### Veri Temizleme
Eksik verilerin mantıksal yöntemlerle doldurulması ve veri bütünlüğü kontrolleri gerçekleştirildi.

### Özellik Mühendisliği
Mevcut sütunlardan misafir profili, döngüsel zaman özellikleri, finansal metrikler ve rezervasyon kalitesi göstergeleri türetildi.

### Aykırı Değer Yönetimi
Modelin kararlılığını etkileyebilecek uç değerler kontrollü biçimde baskılandı.

### İstatistiksel Doğrulama
Yeni türetilen özelliklerin hedef değişken (`is_canceled`) üzerindeki etkisinin istatistiksel anlamda doğrulanması sağlandı.

---

## Özellik Mühendisliği Uygulamaları

### Zaman Analizi

`arrival_date` değişkeni bileşenlerine ayrıştırılarak yıl, ay ve gün bilgileri yapılandırılmıştır. Ayın döngüsel doğasını korumak için sinüs ve kosinüs dönüşümleri uygulanmıştır. Varış gününün hafta sonu olup olmadığını gösteren `is_weekend` değişkeni oluşturulmuştur.

### Misafir Profili Segmentasyonu

Misafirlerin grup yapısını analiz etmek için `is_family`, `is_single` ve `is_couple` kategorileri tanımlanmıştır. Toplam konaklama süresi `total_stay` değişkeni ile hesaplanmış, hafta içi ve hafta sonu konaklama süreleri birleştirilerek daha kapsamlı bir metrik elde edilmiştir.

### Rezervasyon Kalitesi Göstergeleri

`room_type_match` değişkeni (rezerve edilen oda tipi ile atanan oda tipi eşleşmesi) müşteri memnuniyetinin önemli bir göstergesidir. `lead_time` değişkeni kategorik gruplara (`lead_time_bins`) bölünerek model için daha anlamlı hale getirilmiştir.

### Finansal Metrikler

Misafirlerin harcama kapasitesi `price_per_person` değişkeni ile belirlenmiş, fiyat verisindeki sağa çarpıklığı azaltmak için `adr_log` (oda fiyatının logaritmik dönüşümü) uygulanmıştır.

---

## Temel Bulgular

**Oda Tipi Uyumsuzluğu:** Talep ettiği oda tipi verilmeyen misafirlerin iptal etme olasılığı, istediği odayı alanlarla karşılaştırıldığında anlamlı derecede daha yüksektir.

**Lead Time ve İptal İlişkisi:** Rezervasyon tarihi ile varış tarihi arasındaki süre arttıkça, iptal olasılığı doğrusal olarak artmaktadır.

**Geçmiş Davranış Örüntüleri:** Daha önceden iptal yapmış olan misafirlerin (`previous_cancel_ratio`), tekrar iptal etme eğilimi gözlemlenmiştir. Bu değişken, modelin en güçlü tahminleyicileri arasında yer almaktadır.

---

## Kullanılan Teknolojiler

| Teknoloji | Amaç |
|-----------|------|
| Python 3.x | Genel programlama dili |
| Pandas | Veri işleme ve manipülasyonu |
| NumPy | Sayısal hesaplamalar |
| Matplotlib | Veri görselleştirmesi |
| Seaborn | İstatistiksel görselleştirme |
 

### Gerekli İçe Aktarmalar

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
import warnings
```

---

## Özellik Önem Analizi

1. **previous_cancel_ratio** - Misafirlerin geçmiş iptal davranışı
2. **lead_time** - Rezervasyon öncesi geçen süre
3. **adr_log** - Logaritmik dönüşüm uygulanmış oda fiyatı

---

## Sonuç

Proje kapsamında gerçekleştirilen veri temizleme ve özellik mühendisliği çalışmaları, ham verileri makine öğrenmesi modellerinin etkili bir şekilde öğrenebileceği nitelikli özelliklere dönüştürmüştür. Elde edilen bulgular, otel işletmeciliğinde müşteri iptal davranışlarını etkileyen faktörlerin doğru şekilde tanımlanması ve iş stratejilerine entegre edilmesi için önemli içgörüler sunmaktadır.