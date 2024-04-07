import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.linear_model import ElasticNet, BayesianRidge, HuberRegressor, SGDRegressor, Ridge, Lasso, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.mixture import GaussianMixture
from sklearn.kernel_ridge import KernelRidge
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import PairwiseKernel

warnings.filterwarnings('ignore')


# Step 1: Load the Dataset
diamond_data = pd.read_csv('diamonds.csv')

# Step 2: Data Cleaning

# Handling missing values (if any)
diamond_data.dropna(inplace=True)

# Step 3: Preprocess the Data
label_encoders = {}
for column in ['cut', 'color', 'clarity']:
    label_encoders[column] = LabelEncoder()
    diamond_data[column] = label_encoders[column].fit_transform(diamond_data[column])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(diamond_data.drop(columns=['price']))


# Step 4: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, diamond_data['price'], test_size=0.2, random_state=42)

# Step 5: Train and Evaluate Models

# Support Vector Machine (SVM)
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)

svm_mae = mean_absolute_error(y_test, svm_predictions)
svm_rmse = mean_squared_error(y_test, svm_predictions, squared=False)
svm_r2 = r2_score(y_test, svm_predictions)

svm_accuracy = sum(abs(y_test - svm_predictions) <= 5) / len(y_test) * 100

print("Support Vector Machine:")
print("MAE:", svm_mae)
print("RMSE:", svm_rmse)
print("R-squared:", svm_r2)
print("Accuracy (within 5%): {:.2f}%".format(svm_accuracy))

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)
rf_r2 = r2_score(y_test, rf_predictions)

rf_accuracy = sum(abs(y_test - rf_predictions) <= 5) / len(y_test) * 100

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R-squared:", rf_r2)
print("Accuracy (within 5%): {:.2f}%".format(rf_accuracy))

# Decision Trees
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_rmse = mean_squared_error(y_test, dt_predictions, squared=False)
dt_r2 = r2_score(y_test, dt_predictions)

dt_accuracy = sum(abs(y_test - dt_predictions) <= 5) / len(y_test) * 100

print("\nDecision Trees:")
print("MAE:", dt_mae)
print("RMSE:", dt_rmse)
print("R-squared:", dt_r2)
print("Accuracy (within 5%): {:.2f}%".format(dt_accuracy))

# Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_predictions)
gb_rmse = mean_squared_error(y_test, gb_predictions, squared=False)
gb_r2 = r2_score(y_test, gb_predictions)

gb_accuracy = sum(abs(y_test - gb_predictions) <= 5) / len(y_test) * 100

print("\nGradient Boosting:")
print("MAE:", gb_mae)
print("RMSE:", gb_rmse)
print("R-squared:", gb_r2)
print("Accuracy (within 5%): {:.2f}%".format(gb_accuracy))

# AdaBoost
ab_model = AdaBoostRegressor(random_state=42)
ab_model.fit(X_train, y_train)
ab_predictions = ab_model.predict(X_test)

ab_mae = mean_absolute_error(y_test, ab_predictions)
ab_rmse = mean_squared_error(y_test, ab_predictions, squared=False)
ab_r2 = r2_score(y_test, ab_predictions)

ab_accuracy = sum(abs(y_test - ab_predictions) <= 5) / len(y_test) * 100

print("\nAdaBoost:")
print("MAE:", ab_mae)
print("RMSE:", ab_rmse)
print("R-squared:", ab_r2)
print("Accuracy (within 5%): {:.2f}%".format(ab_accuracy))

# Bagging
bg_model = BaggingRegressor(random_state=42)
bg_model.fit(X_train, y_train)
bg_predictions = bg_model.predict(X_test)

bg_mae = mean_absolute_error(y_test, bg_predictions)
bg_rmse = mean_squared_error(y_test, bg_predictions, squared=False)
bg_r2 = r2_score(y_test, bg_predictions)

bg_accuracy = sum(abs(y_test - bg_predictions) <= 5) / len(y_test) * 100

print("\nBagging:")
print("MAE:", bg_mae)
print("RMSE:", bg_rmse)
print("R-squared:", bg_r2)
print("Accuracy (within 5%): {:.2f}%".format(bg_accuracy))

# K-Nearest Neighbors (KNN)
knn_model = KNeighborsRegressor()
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

knn_mae = mean_absolute_error(y_test, knn_predictions)
knn_rmse = mean_squared_error(y_test, knn_predictions, squared=False)
knn_r2 = r2_score(y_test, knn_predictions)

knn_accuracy = sum(abs(y_test - knn_predictions) <= 5) / len(y_test) * 100

print("\nK-Nearest Neighbors (KNN):")
print("MAE:", knn_mae)
print("RMSE:", knn_rmse)
print("R-squared:", knn_r2)
print("Accuracy (within 5%): {:.2f}%".format(knn_accuracy))

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = mean_squared_error(y_test, lr_predictions, squared=False)
lr_r2 = r2_score(y_test, lr_predictions)

lr_accuracy = sum(abs(y_test - lr_predictions) <= 5) / len(y_test) * 100

print("\nLinear Regression:")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)
print("R-squared:", lr_r2)
print("Accuracy (within 5%): {:.2f}%".format(lr_accuracy))

# Ridge Regression
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_rmse = mean_squared_error(y_test, ridge_predictions, squared=False)
ridge_r2 = r2_score(y_test, ridge_predictions)

ridge_accuracy = sum(abs(y_test - ridge_predictions) <= 5) / len(y_test) * 100

print("\nRidge Regression:")
print("MAE:", ridge_mae)
print("RMSE:", ridge_rmse)
print("R-squared:", ridge_r2)
print("Accuracy (within 5%): {:.2f}%".format(ridge_accuracy))

# Lasso Regression
lasso_model = Lasso()
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

lasso_mae = mean_absolute_error(y_test, lasso_predictions)
lasso_rmse = mean_squared_error(y_test, lasso_predictions, squared=False)
lasso_r2 = r2_score(y_test, lasso_predictions)

lasso_accuracy = sum(abs(y_test - lasso_predictions) <= 5) / len(y_test) * 100

print("\nLasso Regression:")
print("MAE:", lasso_mae)
print("RMSE:", lasso_rmse)
print("R-squared:", lasso_r2)
print("Accuracy (within 5%): {:.2f}%".format(lasso_accuracy))


# ElasticNet Regression
enet_model = ElasticNet()
enet_model.fit(X_train, y_train)
enet_predictions = enet_model.predict(X_test)

enet_mae = mean_absolute_error(y_test, enet_predictions)
enet_rmse = mean_squared_error(y_test, enet_predictions, squared=False)
enet_r2 = r2_score(y_test, enet_predictions)

enet_accuracy = sum(abs(y_test - enet_predictions) <= 5) / len(y_test) * 100

print("\nElasticNet Regression:")
print("MAE:", enet_mae)
print("RMSE:", enet_rmse)
print("R-squared:", enet_r2)
print("Accuracy (within 5%): {:.2f}%".format(enet_accuracy))

# Bayesian Ridge Regression
bayesian_model = BayesianRidge()
bayesian_model.fit(X_train, y_train)
bayesian_predictions = bayesian_model.predict(X_test)

bayesian_mae = mean_absolute_error(y_test, bayesian_predictions)
bayesian_rmse = mean_squared_error(y_test, bayesian_predictions, squared=False)
bayesian_r2 = r2_score(y_test, bayesian_predictions)

bayesian_accuracy = sum(abs(y_test - bayesian_predictions) <= 5) / len(y_test) * 100

print("\nBayesian Ridge Regression:")
print("MAE:", bayesian_mae)
print("RMSE:", bayesian_rmse)
print("R-squared:", bayesian_r2)
print("Accuracy (within 5%): {:.2f}%".format(bayesian_accuracy))

# Huber Regressor
huber_model = HuberRegressor()
huber_model.fit(X_train, y_train)
huber_predictions = huber_model.predict(X_test)

huber_mae = mean_absolute_error(y_test, huber_predictions)
huber_rmse = mean_squared_error(y_test, huber_predictions, squared=False)
huber_r2 = r2_score(y_test, huber_predictions)

huber_accuracy = sum(abs(y_test - huber_predictions) <= 5) / len(y_test) * 100

print("\nHuber Regressor:")
print("MAE:", huber_mae)
print("RMSE:", huber_rmse)
print("R-squared:", huber_r2)
print("Accuracy (within 5%): {:.2f}%".format(huber_accuracy))

# SGD Regressor
sgd_model = SGDRegressor()
sgd_model.fit(X_train, y_train)
sgd_predictions = sgd_model.predict(X_test)

sgd_mae = mean_absolute_error(y_test, sgd_predictions)
sgd_rmse = mean_squared_error(y_test, sgd_predictions, squared=False)
sgd_r2 = r2_score(y_test, sgd_predictions)

sgd_accuracy = sum(abs(y_test - sgd_predictions) <= 5) / len(y_test) * 100

print("\nSGD Regressor:")
print("MAE:", sgd_mae)
print("RMSE:", sgd_rmse)
print("R-squared:", sgd_r2)
print("Accuracy (within 5%): {:.2f}%".format(sgd_accuracy))


# XGBoost Regressor
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
xgb_r2 = r2_score(y_test, xgb_predictions)
xgb_accuracy = sum(abs(y_test - xgb_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nXGBoost Regressor:")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R-squared:", xgb_r2)
print("Accuracy (within 5%): {:.2f}%".format(xgb_accuracy))

# LightGBM Regressor
lgb_model = LGBMRegressor()
lgb_model.fit(X_train, y_train)
lgb_predictions = lgb_model.predict(X_test)

lgb_mae = mean_absolute_error(y_test, lgb_predictions)
lgb_rmse = mean_squared_error(y_test, lgb_predictions, squared=False)
lgb_r2 = r2_score(y_test, lgb_predictions)
lgb_accuracy = sum(abs(y_test - lgb_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nLightGBM Regressor:")
print("MAE:", lgb_mae)
print("RMSE:", lgb_rmse)
print("R-squared:", lgb_r2)
print("Accuracy (within 5%): {:.2f}%".format(lgb_accuracy))

# CatBoost Regressor
catboost_model = CatBoostRegressor()
catboost_model.fit(X_train, y_train)
catboost_predictions = catboost_model.predict(X_test)

catboost_mae = mean_absolute_error(y_test, catboost_predictions)
catboost_rmse = mean_squared_error(y_test, catboost_predictions, squared=False)
catboost_r2 = r2_score(y_test, catboost_predictions)
catboost_accuracy = sum(abs(y_test - catboost_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nCatBoost Regressor:")
print("MAE:", catboost_mae)
print("RMSE:", catboost_rmse)
print("R-squared:", catboost_r2)
print("Accuracy (within 5%): {:.2f}%".format(catboost_accuracy))

# Neural Network
nn_model = MLPRegressor(random_state=42)
nn_model.fit(X_train, y_train)
nn_predictions = nn_model.predict(X_test)

nn_mae = mean_absolute_error(y_test, nn_predictions)
nn_rmse = mean_squared_error(y_test, nn_predictions, squared=False)
nn_r2 = r2_score(y_test, nn_predictions)
nn_accuracy = sum(abs(y_test - nn_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nNeural Network:")
print("MAE:", nn_mae)
print("RMSE:", nn_rmse)
print("R-squared:", nn_r2)
print("Accuracy (within 5%): {:.2f}%".format(nn_accuracy))


# RANSAC Regressor
ransac_model = RANSACRegressor(random_state=42)
ransac_model.fit(X_train, y_train)
ransac_predictions = ransac_model.predict(X_test)

ransac_mae = mean_absolute_error(y_test, ransac_predictions)
ransac_rmse = mean_squared_error(y_test, ransac_predictions, squared=False)
ransac_r2 = r2_score(y_test, ransac_predictions)
ransac_accuracy = sum(abs(y_test - ransac_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nRANSAC Regressor:")
print("MAE:", ransac_mae)
print("RMSE:", ransac_rmse)
print("R-squared:", ransac_r2)
print("Accuracy (within 5%): {:.2f}%".format(ransac_accuracy))

# Gaussian Mixture Model
gmm_model = GaussianMixture()
gmm_model.fit(X_train, y_train)
gmm_predictions = gmm_model.predict(X_test)

gmm_mae = mean_absolute_error(y_test, gmm_predictions)
gmm_rmse = mean_squared_error(y_test, gmm_predictions, squared=False)
gmm_r2 = r2_score(y_test, gmm_predictions)
gmm_accuracy = sum(abs(y_test - gmm_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nGaussian Mixture Model:")
print("MAE:", gmm_mae)
print("RMSE:", gmm_rmse)
print("R-squared:", gmm_r2)
print("Accuracy (within 5%): {:.2f}%".format(gmm_accuracy))

# Kernel Ridge Regression
kernel_ridge_model = KernelRidge()
kernel_ridge_model.fit(X_train, y_train)
kernel_ridge_predictions = kernel_ridge_model.predict(X_test)

kernel_ridge_mae = mean_absolute_error(y_test, kernel_ridge_predictions)
kernel_ridge_rmse = mean_squared_error(y_test, kernel_ridge_predictions, squared=False)
kernel_ridge_r2 = r2_score(y_test, kernel_ridge_predictions)
kernel_ridge_accuracy = sum(abs(y_test - kernel_ridge_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nKernel Ridge Regression:")
print("MAE:", kernel_ridge_mae)
print("RMSE:", kernel_ridge_rmse)
print("R-squared:", kernel_ridge_r2)
print("Accuracy (within 5%): {:.2f}%".format(kernel_ridge_accuracy))

# Transformed Target Regression
transformed_model = TransformedTargetRegressor()
transformed_model.fit(X_train, y_train)
transformed_predictions = transformed_model.predict(X_test)

transformed_mae = mean_absolute_error(y_test, transformed_predictions)
transformed_rmse = mean_squared_error(y_test, transformed_predictions, squared=False)
transformed_r2 = r2_score(y_test, transformed_predictions)
transformed_accuracy = sum(abs(y_test - transformed_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nTransformed Target Regression:")
print("MAE:", transformed_mae)
print("RMSE:", transformed_rmse)
print("R-squared:", transformed_r2)
print("Accuracy (within 5%): {:.2f}%".format(transformed_accuracy))

# Quantile Transformer
quantile_model = make_pipeline(QuantileTransformer(), LinearRegression())
quantile_model.fit(X_train, y_train)
quantile_predictions = quantile_model.predict(X_test)

quantile_mae = mean_absolute_error(y_test, quantile_predictions)
quantile_rmse = mean_squared_error(y_test, quantile_predictions, squared=False)
quantile_r2 = r2_score(y_test, quantile_predictions)
quantile_accuracy = sum(abs(y_test - quantile_predictions) <= 0.05 * y_test) / len(y_test) * 100

print("\nQuantile Transformer:")
print("MAE:", quantile_mae)
print("RMSE:", quantile_rmse)
print("R-squared:", quantile_r2)
print("Accuracy (within 5%): {:.2f}%".format(quantile_accuracy))
