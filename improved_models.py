import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load the Dataset
diamond_data = pd.read_csv('diamonds.csv')

# Step 2: Data Cleaning and Preprocessing

# Handling missing values (if any)
diamond_data.dropna(inplace=True)

# Outlier Detection and Removal
def remove_outliers(df, features):
    z_scores = np.abs((df[features] - df[features].mean()) / df[features].std())
    df = df[(z_scores < 3).all(axis=1)]
    return df

numerical_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
diamond_data = remove_outliers(diamond_data, numerical_features)

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
svm_params = {'C': [0.1, 1, 10], 'gamma': [0.01, 0.1, 1], 'epsilon': [0.1, 0.5, 1]}
svm_grid = GridSearchCV(SVR(kernel='rbf'), svm_params, cv=5)
svm_grid.fit(X_train, y_train)
svm_model = svm_grid.best_estimator_
svm_predictions = svm_model.predict(X_test)

svm_mae = mean_absolute_error(y_test, svm_predictions)
svm_rmse = mean_squared_error(y_test, svm_predictions, squared=False)
svm_r2 = r2_score(y_test, svm_predictions)

svm_accuracy = sum(abs(y_test - svm_predictions) <= 5 * y_test) / len(y_test) * 100

print("Support Vector Machine:")
print("MAE:", svm_mae)
print("RMSE:", svm_rmse)
print("R-squared:", svm_r2)
print("Accuracy (within 5%): {:.2f}%".format(svm_accuracy))

# Decision Trees
dt_params = {'max_depth': [None, 10, 50, 100]}
dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=5)
dt_grid.fit(X_train, y_train)
dt_model = dt_grid.best_estimator_
dt_predictions = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_rmse = mean_squared_error(y_test, dt_predictions, squared=False)
dt_r2 = r2_score(y_test, dt_predictions)

dt_accuracy = sum(abs(y_test - dt_predictions) <= 5 * y_test) / len(y_test) * 100

print("\nDecision Trees:")
print("MAE:", dt_mae)
print("RMSE:", dt_rmse)
print("R-squared:", dt_r2)
print("Accuracy (within 5%): {:.2f}%".format(dt_accuracy))

# Random Forest
rf_params = {'n_estimators': [100, 500], 'max_depth': [None, 10, 50, 100]}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
rf_grid.fit(X_train, y_train)
rf_model = rf_grid.best_estimator_
rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions, squared=False)
rf_r2 = r2_score(y_test, rf_predictions)

rf_accuracy = sum(abs(y_test - rf_predictions) <= 5 * y_test) / len(y_test) * 100

print("\nRandom Forest:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R-squared:", rf_r2)
print("Accuracy (within 5%): {:.2f}%".format(rf_accuracy))

# Bagging
bg_params = {'n_estimators': [10, 50, 100]}
bg_grid = GridSearchCV(BaggingRegressor(random_state=42), bg_params, cv=5)
bg_grid.fit(X_train, y_train)
bg_model = bg_grid.best_estimator_
bg_predictions = bg_model.predict(X_test)

bg_mae = mean_absolute_error(y_test, bg_predictions)
bg_rmse = mean_squared_error(y_test, bg_predictions, squared=False)
bg_r2 = r2_score(y_test, bg_predictions)

bg_accuracy = sum(abs(y_test - bg_predictions) <= 5 * y_test) / len(y_test) * 100

print("\nBagging:")
print("MAE:", bg_mae)
print("RMSE:", bg_rmse)
print("R-squared:", bg_r2)
print("Accuracy (within 5%): {:.2f}%".format(bg_accuracy))

# XGBoost
xgb_params = {'n_estimators': [100, 500], 'max_depth': [3, 6, 9]}
xgb_grid = GridSearchCV(XGBRegressor(random_state=42), xgb_params, cv=5)
xgb_grid.fit(X_train, y_train)
xgb_model = xgb_grid.best_estimator_
xgb_predictions = xgb_model.predict(X_test)

xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_rmse = mean_squared_error(y_test, xgb_predictions, squared=False)
xgb_r2 = r2_score(y_test, xgb_predictions)

xgb_accuracy = sum(abs(y_test - xgb_predictions) <= 5 * y_test) / len(y_test) * 100

print("\nXGBoost:")
print("MAE:", xgb_mae)
print("RMSE:", xgb_rmse)
print("R-squared:", xgb_r2)
print("Accuracy (within 5%): {:.2f}%".format(xgb_accuracy))
