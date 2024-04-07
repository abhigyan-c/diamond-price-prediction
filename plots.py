import matplotlib.pyplot as plt

# Data
models = ['SVM', 'Random Forest', 'Decision Trees', 'Gradient Boosting', 'AdaBoost', 'Bagging', 
          'KNN', 'Linear Regression', 'Ridge Regression', 'Lasso Regression', 'ElasticNet Regression', 
          'Bayesian Ridge Regression', 'Huber Regressor', 'SGD Regressor', 'XGBoost', 'LightGBM', 
          'CatBoost', 'Neural Network', 'Kernel Ridge', 'Transformed Target', 'Quantile Transformer']

mae = [1313.29, 2.75, 2.23, 127.63, 340.56, 2.91, 316.54, 862.99, 863.09, 864.11, 1120.07, 863.05, 763.79, 
       874.85, 26.28, 29.70, 39.11, 397.64, 3944.93, 862.99, 1579.32]

rmse = [2762.49, 21.23, 22.76, 202.43, 463.21, 19.94, 665.32, 1346.11, 1346.11, 1346.24, 1670.97, 1346.11, 
        1417.68, 1351.83, 55.41, 50.90, 62.88, 708.50, 4144.84, 1346.11, 2100.26]

r_squared = [0.52, 0.99997, 0.99997, 0.99742, 0.98650, 0.99997, 0.97215, 0.88601, 0.88601, 0.88601, 0.82436, 
             0.88601, 0.87357, 0.88504, 0.99981, 0.99984, 0.99975, 0.96842, -0.08070, 0.88601, 0.72252]

accuracy = [0.64, 93.47, 96.36, 2.48, 0.91, 94.30, 5.22, 0.53, 0.52, 0.45, 0.38, 0.52, 0.81, 0.46, 94.00, 90.16, 
            88.95, 29.51, 0.24, 12.64, 5.04]

# Plotting
plt.figure(figsize=(8, 6))
plt.barh(models, mae, color='skyblue')
plt.xlabel('MAE')
plt.title('Mean Absolute Error')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.barh(models, rmse, color='lightgreen')
plt.xlabel('RMSE')
plt.title('Root Mean Squared Error')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.barh(models, r_squared, color='salmon')
plt.xlabel('R-squared')
plt.title('R-squared')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.barh(models, accuracy, color='gold')
plt.xlabel('Accuracy (%)')
plt.title('Accuracy')
plt.tight_layout()
plt.show()
