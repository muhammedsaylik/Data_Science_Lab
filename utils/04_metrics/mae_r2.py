from sklearn.metrics import mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")