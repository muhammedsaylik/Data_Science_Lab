import joblib
import os
from datetime import datetime

def save_model(model, model_name, path="04_Machine_Learning/house-price-prediction/models/"):
    if not os.path.exists(path):
        os.makedirs(path)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{model_name}_{timestamp}.pkl"
    full_path = os.path.join(path, filename)
    
    joblib.dump(model, full_path)
    print(f"Model başarıyla kaydedildi: {full_path}")