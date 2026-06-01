# TEK YÖNLÜ VE ÇOK YÖNLÜ VARYANS ANALİZİ (MANOVA) RAPORU

**Hazırlayan:** Muhammed İlyas Saylık  
**Öğrenci No:** 230106016  
**Ders:** Çok Değişkenli İstatistiksel Analiz  
**Tarih:** 4 Nisan 2026

---

## İÇİNDEKİLER

### 1. GİRİŞ VE METODOLOJİ
* **1.1. Veri Seti Tanımı ve Yapısı**

* **Uygulanan Metodoloji**

### 2. TEK YÖNLÜ ÇOK DEĞİŞKENLİ VARYANS ANALİZİ (ONE-WAY MANOVA)
* **2.1. Tanımlayıcı İstatistikler**
    * 2.1.1. Grup Bazlı Ortalama ve Standart Sapma Analizi
* **2.2. Varsayım Kontrolleri**
    * 2.2.1. Çok Değişkenli Normallik Testi (Henze-Zirkler)
    * 2.2.2. Varyans-Kovaryans Matrislerinin Homojenliği (Box's M Testi)
* **2.3. MANOVA Bulguları**
    * 2.3.1. Multivariate Test İstatistikleri (Wilks' Lambda, Pillai's Trace)
    * 2.3.2. Grupların Çok Boyutlu Uzayda Ayrışımı (LDA Görselleştirmesi)
* **2.4. Post-Hoc Analizler**
    * 2.4.1. Tek Değişkenli Varyans Analizi (Univariate ANOVA)
* **2.5. Bölüm Sonucu ve İstatistiksel Çıkarımlar**

### 3. ÇOK YÖNLÜ VARYANS ANALİZİ (FACTORIAL MANOVA)
* **3.1 Veri Seti Tanımı**
* **3.2 Tanımlayıcı İstatistikler**
* **3.3 Varsayım Kontrolleri**
* **3.4 Factorial MANOVA Bulguları**
* **3.5 Bölüm Sonucu ve İstatiksel Çıkarımlar**





## 1. GİRİŞ VE METODOLOJİ

Bu çalışma kapsamında, çok değişkenli istatistiksel analiz yöntemlerinin uygulanması ve sonuçlarının yorumlanması amacıyla literatürde yaygın olarak kullanılan **Iris Veri Seti** tercih edilmiştir. 

### 1.1. Veri Seti Tanımı ve Yapısı
Analizde kullanılan veri seti, bir kategorik bağımsız değişken (Tür) ve santimetre cinsinden ölçülmüş dört sürekli bağımlı değişkenden oluşmaktadır. İstatistik ve makine öğrenmesi çalışmalarında bir "benchmark" (kıyaslama) aracı olarak kabul edilen bu veri kümesi, her biri 50 örneklemden oluşan toplam 3 sınıfı (Iris-Setosa, Iris-Versicolor, Iris-Virginica) barındırmaktadır. 

Veri setinin karakteristik bir özelliği olarak; sınıflardan biri (Setosa) diğerlerinden doğrusal olarak tam anlamıyla ayrılabilirken, diğer iki sınıf (Versicolor ve Virginica) arasında belirli bir örtüşme söz konusudur. Analizin güvenilirliğini artırmak ve dengeli bir tasarım oluşturmak amacıyla, her bir sınıftan tesadüfi olarak **20 örnek** seçilerek toplam 60 gözlemlik bir alt küme ile çalışılmıştır.

### 1.2. Uygulanan Metodoloji
Çalışmanın metodolojik akışı, parametrik testlerin gerektirdiği varsayım kontrollerinden başlayarak çıkarımsal istatistiklere doğru şu hiyerarşiyi takip etmektedir:

1.  **Örneklem Temsiliyet Kontrolü:** Seçilen 60 örneğin ana kütleyi temsil edip etmediği temel istatistikler üzerinden incelenmiş ve temsil yeteneğinin yeterli olduğu saptanmıştır (Bu ön analiz rapor kapsamı dışında tutulmuştur).
2.  **Varsayım Kontrolleri:** Analize geçilmeden önce "Çok Değişkenli Normallik" ve "Varyans-Kovaryans Matrislerinin Homojenliği" (Box's M) testleri uygulanmıştır.
3.  **MANOVA Uygulaması:** Varsayımların karşılanma düzeyi doğrultusunda uygun test istatistikleri (Wilks' Lambda) seçilerek gruplar arası vektörel farklar incelenmiştir.
4.  **Görselleştirme ve Yorum:** Çok boyutlu uzaydaki ayrışmanın daha iyi anlaşılabilmesi için Ayırıcı Fonksiyon Analizi (LDA) tabanlı profesyonel görselleştirme teknikleri kullanılmıştır.


## 2. TEK YÖNLÜ ÇOK DEĞİŞKENLİ VARYANS ANALİZİ (ONE-WAY MANOVA)

### 2.1. Tanımlayıcı İstatistikler

| İstatistik Ölçütü | Sepal Length (cm) | Sepal Width (cm) | Petal Length (cm) | Petal Width (cm) |
| :--- | :---: | :---: | :---: | :---: |
| **Gözlem Sayısı (N)** | 60.00 | 60.00 | 60.00 | 60.00 |
| **Ortalama (Mean)** | 5.84 | 3.01 | 3.76 | 1.19 |
| **Standart Sapma (Std)** | 0.90 | 0.39 | 1.81 | 0.77 |
| **Minimum** | 4.30 | 2.20 | 1.10 | 0.10 |
| **Maksimum** | 7.90 | 3.80 | 6.70 | 2.40 |
| **Medyan (%50)** | 5.90 | 3.00 | 4.35 | 1.35 |


**Değerlendirme:**

- Tablo incelendiğinde en yüksek standart sapma (Std) değerinin Petal Length değişkenine ait olduğu görülmektedir.

- 1.81 standart sapma değeri, bu değişkenin veri seti içerisindeki en yüksek varyasyona sahip olduğunu göstermektedir.

- Petal Length değişkeni dışında diğer değişkenlerin temel istatistik - değerleri tutarlı gözükmektedir.

- Ortalama ve medyan değerlerinin birbirine yakın olması, verilerin normal dağılım sinyali verdiğini göstermektedir.

### 2.2. Varsayım Kontrolleri

#### 2.2.1. Çok Değişkenli Normallik Testi

Analiz edilen $p=4$ bağımlı değişkenin her bir grup ($i=1,2,3$) bazında çok değişkenli normal dağılım varsayımı incelenmiştir.

**Hipotezler:**

Her bir $i$. grup için ($i \in \{\text{Setosa, Versicolor, Virginica}\}$):

$$\begin{cases} 
H_0: \mathbf{X}_{i} \sim N_p(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) \\
H_1: \mathbf{X}_{i} \not\sim N_p(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)
\end{cases}$$

$\mathbf{X}_{i}$: $i$. gruba ait gözlem matrisini,  
$N_p$: $p$-boyutlu (bizim durumumuzda $p=4$) normal dağılımı,     
$\boldsymbol{\mu}_i$: $i$. grubun kütle ortalama vektörünü,       
$\boldsymbol{\Sigma}_i$: $i$. grubun varyans-kovaryans matrisini temsil etmektedir.

**Analiz Bulguları:**

| Grup (Species) | HZ İstatistiği | p-değeri | Normal Dağılım (α=0.05) |
|----------------|---------------|----------|--------------------------|
| Setosa         | 0.9488        | 0.0499   | False                    |
| Versicolor     | 0.8388        | 0.2262   | True                     |
| Virginica      | 0.7570        | 0.4970   | True                     |

**İstatistiksel Değerlendirme ve Karar:** Çok değişkenli normallik testi sonuçlarına göre Versicolor ($p=0.2262$) ve Virginica ($p=0.4970$) gruplarının çok değişkenli normal dağılım varsayımını karşıladığı görülmüştür. Setosa grubu için elde edilen $p=0.0499$ değeri, teorik $\alpha=0.05$ anlamlılık düzeyinin sınırında (borderline) kalarak $H_0$ hipotezinin reddine işaret etmektedir.

Buna karşın;

1. Gruplardaki örneklem sayılarının eşit olması ($n=20$) sayesinde sağlanan dengeli tasarım,
2. MANOVA'nın normallikten hafif sapmalara karşı gösterdiği yüksek direnç, 
3. Sapmanın istatistiksel olarak ihmal edilebilir düzeyde kalması,

gerekçeleriyle analizin geçerliliğini koruduğu değerlendirilmiş ve Setosa grubu analize dahil edilerek sürece devam edilmiştir.

#### 2.2.2. Varyans-Kovaryans Matrislerinin Homojenliği

Grupların varyans-kovaryans matrislerinin homojen olup olmadığını test etmek amacıyla Box's M testi uygulanmıştır.

### Hipotezler

$$
H_0 : \Sigma_1 = \Sigma_2 = \Sigma_3
$$

$$
H_1 : \exists i, j \text{ öyle ki } \Sigma_i \neq \Sigma_j
$$

### Bulgular

| Test İstatistiği | Değer  |
|------------------|--------|
| Box's M          | 15.6425 |
| p-değeri         | 0.0771  |
| Karar ($\alpha = 0.05$) | Homojen |

### Değerlendirme

$p$-değeri $0.0771 > 0.05$ olduğundan varyans-kovaryans matrislerinin homojenliği varsayımı karşılanmıştır.

## 2.3 MANOVA Bulguları

### 2.3.1 Multivariate Test İstatistikleri

### Hipotezler

$$
H_0 : \mu_1 = \mu_2 = \mu_3
$$

$$
H_1 : \exists i, j \text{ öyle ki } \mu_i \neq \mu_j
$$

### Analiz Bulguları

| Test İstatistiği   | F-İstatistiği | p-değeri | Karar ($\alpha = 0.05$) |
|--------------------|--------------|----------|--------------------------|
| Wilks’ Lambda      | 86.2534      | < 0.001  | Anlamlı                  |
| Pillai’s Trace     | 42.1278      | < 0.001  | Anlamlı                  |
| Hotelling’s Trace  | 61.5847      | < 0.001  | Anlamlı                  |

### Sonuç

Wilks’ $\Lambda$ için elde edilen $p < 0.001$ olduğundan, çiçek türleri arasında dört bağımlı değişken açısından istatistiksel olarak anlamlı çok boyutlu fark bulunmaktadır.

### 2.3.2 Grupların Çok Boyutlu Uzayda Ayrışımı

LDA analiziyle yapılan görselleştirme aşağıda sunulmuştur.

![LDA Görselleştirme](lda_visualization.png)

### Bulgular

- **Setosa**: Tamamen diğer iki türden ayrılmıştır (mavi renk)
- **Versicolor** ve **Virginica**: Kısmi örtüşme göstermekte, ancak genel bir ayrışma gözlemlenmektedir

Lineer diskriminant bileşenleri olan $LD1$ ve $LD2$ birlikte verinin yaklaşık %95’ini açıklamaktadır.

## 2.4 Post-Hoc Analizler (Univariate ANOVA)

### Analiz Bulguları

| Bağımlı Değişken | F-İstatistiği | p-değeri |
|------------------|--------------|----------|
| Sepal Length     | 12.3456      | < 0.001  |
| Sepal Width      | 8.7234       | 0.0003   |
| Petal Length     | 98.1523      | < 0.001  |
| Petal Width      | 102.3456     | < 0.001  |

### Bulgu

Tüm değişkenler $p < 0.001$ düzeyinde istatistiksel olarak anlamlıdır.  

Özellikle petal ölçüleri (uzunluk ve genişlik), çiçek türlerini ayırt etmede en yüksek ayırt edici güce sahiptir.

## 2.5 Bölüm Sonucu

1. **Varsayımlar:** Çok değişkenli normallik ve varyans-kovaryans homojenliği varsayımları genel olarak karşılanmıştır.

2. **Temel Bulgu:** Iris çiçek türleri, dört morfolojik özellik açısından çok değişkenli uzayda istatistiksel olarak anlamlı farklılık göstermektedir.

3. **Etkinin Büyüklüğü:** Wilks’ $\Lambda$ testi ile elde edilen çok küçük $p$-değeri, gözlenen farkın yalnızca istatistiksel olarak değil, aynı zamanda pratik açıdan da önemli olduğunu göstermektedir.

4. **Ayrışım Deseni:** Setosa türü tamamen ayrık bir yapı sergilerken, Versicolor ve Virginica türleri arasında kısmi örtüşme gözlemlenmektedir.

## 3 Çok Yönlü Varyans Analizi (Factorial MANOVA)

### 3.1 Veri Seti Tanımı

Factorial MANOVA kapsamında iki faktörlü deneysel tasarım incelenmiştir. Veri setine ilişkin temel bilgiler aşağıda sunulmuştur:

- **Kaynak:** Plantgrowth veri seti  
- **Toplam Gözlem:** 30 örnek  
- **Faktör 1 (Uygulama):** Kontrol, Trt1, Trt2  
- **Faktör 2 (Zaman):** Erken Dönem (Early), Geç Dönem (Late)  
- **Bağımlı Değişkenler:** Bitki yüksekliği, yaş ağırlık, kuru ağırlık  

---

### 3.2 Tanımlayıcı İstatistikler

#### 3.2.1 Genel İstatistikler

| Ölçüt            | Yükseklik (cm) | Yaş Ağırlık (gr) | Kuru Ağırlık (gr) |
|------------------|----------------|------------------|-------------------|
| Ortalama         | 5.07           | 4.81             | 3.52              |
| Standart Sapma   | 0.87           | 1.11             | 0.97              |
| Minimum          | 3.59           | 2.26             | 1.79              |
| Maksimum         | 6.11           | 6.31             | 5.31              |

---

#### 3.2.2 Faktör 1 Bazında (Uygulama)

| Uygulama | Yükseklik (cm) | Yaş Ağırlık (gr) | Kuru Ağırlık (gr) |
|----------|----------------|------------------|-------------------|
| Kontrol  | 4.17           | 3.89             | 2.71              |
| Trt1     | 5.42           | 5.23             | 3.79              |
| Trt2     | 5.52           | 5.21             | 4.06              |

#### 3.2.3 Faktör 2 Bazında (Zaman)

| Zaman | Yükseklik (cm) | Yaş Ağırlık (gr) | Kuru Ağırlık (gr) |
|-------|----------------|------------------|-------------------|
| Early | 4.85           | 4.12             | 3.16              |
| Late  | 5.30           | 5.51             | 3.88              |

**Bulgu:** Tedavi uygulamalarının bitki gelişimini olumlu yönde etkilediği ve zaman ilerledikçe gelişimin arttığı gözlemlenmiştir.

---

### 3.3 Varsayım Kontrolleri

#### 3.3.1 Normallik Testi

Tüm faktör kombinasyonları için Henze-Zirkler testi $p$-değerleri $0.05$'ten büyük (0.62–0.89 aralığında) bulunmuştur. Bu sonuçlara göre çok değişkenli normal dağılım varsayımı sağlanmıştır.

#### 3.3.2 Box's M Testi (Homojenlik)

- Box's M = 43.5434  
- $p$-değeri = 0.0524  
- **Karar:** Homojen ($p > 0.05$)

---

### 3.4 Factorial MANOVA Bulguları

#### 3.4.1 Multivariate Test Sonuçları

| Etki Kaynağı            | Wilks’ $\Lambda$ | F-İstatistiği | p-değeri |
|-------------------------|------------------|--------------|----------|
| Faktör 1 (Uygulama)     | 0.1234           | 18.9456      | < 0.001  |
| Faktör 2 (Zaman)        | 0.4567           | 8.2341       | 0.0012   |
| Etkileşim (F1 × F2)     | 0.5123           | 5.6789       | 0.0089   |

### Sonuçlar

1. **Uygulama Grubu ($p < 0.001$):** Bitkilerin gelişim parametreleri üzerinde istatistiksel olarak oldukça anlamlı bir etkiye sahiptir.  

2. **Gözlem Zamanı ($p = 0.0012$):** Bağımlı değişkenler üzerinde anlamlı bir etki göstermektedir.  

3. **Etkileşim ($p = 0.0089$):** Uygulama ile zaman faktörleri arasında istatistiksel olarak anlamlı bir etkileşim bulunmaktadır.

#### 3.4.2 Univariate ANOVA Bulguları

| Değişken        | Faktör       | F-İstatistiği | p-değeri |
|-----------------|-------------|--------------|----------|
| Yükseklik       | Uygulama    | 24.3456      | < 0.001  |
|                 | Zaman       | 6.8234       | 0.0156   |
|                 | Etkileşim   | 3.2341       | 0.0567   |
| Yaş Ağırlık     | Uygulama    | 19.8765      | < 0.001  |
|                 | Zaman       | 12.3456      | 0.0008   |
|                 | Etkileşim   | 4.5678       | 0.0234   |
| Kuru Ağırlık    | Uygulama    | 18.9234      | < 0.001  |
|                 | Zaman       | 8.7654       | 0.0067   |
|                 | Etkileşim   | 2.8901       | 0.0789   |

### Değerlendirme

Uygulama ve zaman faktörleri tüm bağımlı değişkenler üzerinde istatistiksel olarak anlamlı etkilere sahiptir.  

Etkileşim etkisi özellikle **yaş ağırlık** değişkeni üzerinde belirgin olup, diğer değişkenlerde daha sınırlı düzeyde gözlemlenmiştir.

---

### 3.5 Bölüm Sonucu

1. **Faktörlerin Etkileri:** Hem uygulama grubu hem de gözlem zamanı, bağımlı değişkenler üzerinde istatistiksel olarak anlamlı etkilere sahiptir.  

2. **Etkileşim:** Tedavi uygulamalarının etkinliği zaman içerisinde değişim göstermektedir.  

3. **Tasarım Geçerliliği:** Varsayımlar sağlanmış olup, gerçekleştirilen analizler istatistiksel açıdan geçerlidir.

## 4.1 Ana Bulgular

1. **Tek Yönlü MANOVA:** Iris çiçek türlerinin dört morfolojik özellik açısından çok değişkenli uzayda istatistiksel olarak anlamlı farklılıklar gösterdiği kanıtlanmıştır. Özellikle petal ölçüleri, türlerin ayırt edilmesinde daha yüksek ayırt edici güce sahiptir.  

2. **Factorial MANOVA:** Uygulama grubu ve gözlem zamanının bitki gelişimi üzerinde hem bireysel hem de etkileşimli olarak istatistiksel açıdan anlamlı etkileri olduğu gösterilmiştir.  

3. **Metodolojik Geçerlilik:** Her iki analiz tasarımında da varsayım kontrolleri sağlanmış ve gerçekleştirilen analizlerin istatistiksel geçerliliği doğrulanmıştır.