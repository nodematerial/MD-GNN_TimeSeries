import numpy as np
from sklearn.metrics import r2_score

file = 'prediction_with_time.csv'
data = np.loadtxt(file, delimiter=',', skiprows=1)

# Define function to calculate RMSE and R²
def calculate_metrics(start_index, end_index):
    y_true = data[start_index:end_index, 0]
    y_pred = data[start_index:end_index, 2]
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = r2_score(y_true, y_pred)
    return rmse, r2

# Calculate RMSE and R² for each segment
metrics = []
for i in range(10):
    start_index = 21 + i * 300
    end_index = 300 + i * 300
    rmse, r2 = calculate_metrics(start_index, end_index)
    metrics.append((rmse, r2))

# Print RMSE and R² for each segment
for i, (rmse, r2) in enumerate(metrics, 1):
    print(f'rmse{i}: {rmse:.2f}, r2_{i}: {r2:.2f}')
