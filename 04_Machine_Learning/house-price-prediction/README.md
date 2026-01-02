 # 🏠 House Prices Prediction  

Bu proje, **Kaggle House Prices: Advanced Regression Techniques** yarışması için geliştirilmiştir. Amaç, verilen konut özelliklerine göre ev fiyatlarını tahmin etmektir.  

Projede kapsamlı bir **veri ön işleme, özellik mühendisliği, modelleme ve stacking ensemble yaklaşımı** uygulanmıştır.  

---

## 📂 Repository Yapısı  


---

## 🔑 Proje Adımları  

### 1. Veri Keşfi (EDA)  
- Eksik değerlerin analizi (`missingno`, `pandas`)  
- Temel istatistiksel özetler  
- Hedef değişken `SalePrice` dağılımının incelenmesi  

### 2. Veri Ön İşleme  
- Gereksiz sütunların kaldırılması (`MiscFeature, Fence, PoolQC, Alley` vb.)  
- Eksik değerlerin **mod, KNN Imputer ve özel doldurma stratejileri** ile tamamlanması  
- Ordinal değişkenler için **Ordinal Encoding**  
- Kategorik değişkenler için **One-Hot Encoding**  
- Sayısal değişkenler için **RobustScaler** ile ölçeklendirme  

### 3. Modelleme  
- Kullanılan temel modeller:  
  - **GradientBoostingRegressor**  
  - **XGBoost Regressor**  
  - **LightGBM Regressor**  
  - **CatBoost Regressor**  
  - **Ridge, Lasso, ElasticNet, Linear Regression**  

- **Stacking Ensemble** yaklaşımı ile bir **meta-model (Ridge)** üzerinde birleştirilmiştir.  

### 4. Değerlendirme  
Kullanılan metrikler:  
- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² (R-Squared Score)  

#### 📊 Model Sonuçları  
| Model   |   RMSE    |   MAE     |    R²     |
|---------|----------:|----------:|----------:|
| gb      |  0.131303 |  0.086225 | 0.887114  |
| xgb     |  0.127800 |  0.083510 | 0.893368  |
| lgb     |  0.129600 |  0.085152 | 0.890970  |
| cat     |  0.123687 |  0.080917 | 0.900482  |
| ridge   |  0.143798 |  0.088417 | 0.854557  |
| lasso   |  0.139408 |  0.084242 | 0.860449  |
| elastic |  0.140138 |  0.084835 | 0.859695  |
| lr      |  0.146297 |  0.086579 | 0.850324  |
| meta_model (stacking) |  0.122874 | 0.079828 | 0.905312 |
  

---

## 🔄 Akış Diyagramı  

```mermaid
flowchart LR
    %% === DATA ===
    DATA[DATA - train.csv & test.csv]:::data

    %% === full_workflow.ipynb ===
    subgraph WORKFLOW[full_workflow.ipynb]
        direction TB
        EDA[EDA - Scatter, Box, Violin]:::workflow
        subgraph FE[Feature Engineering]
            direction TB
            FE1[Missing Value Handling]:::workflow
            FE2[Encoding]:::workflow
            FE3[Scaling]:::workflow
        end
        MODELS[Ilk Model Denemeleri ve Metric Sonuclari]:::workflow
    end

    %% === model_hub.ipynb ===
    subgraph MODEL[model_hub.ipynb]
        direction TB
        BASE[Base Modeller - GBM, XGB, LGBM, CatBoost]:::model
        META[Stacking Ensemble ve Meta Model Ridge]:::model
        FINAL[Nihai Degerlendirme]:::model
    end

    %% === Connections ===
    DATA --> EDA
    DATA --> FE
    DATA --> MODELS

    EDA --> BASE
    FE --> BASE
    MODELS --> META
    BASE --> META
    META --> FINAL

    %% === Styling ===
    classDef data fill:#87CEEB,stroke:#333,stroke-width:2px,color:#000,font-size:13px;
    classDef workflow fill:#98FB98,stroke:#333,stroke-width:2px,color:#000,font-size:13px;
    classDef model fill:#FFA500,stroke:#333,stroke-width:2px,color:#000,font-size:13px;









