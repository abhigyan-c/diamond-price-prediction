import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

models = {
    "Support Vector Machine": SVR(kernel='rbf'),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Trees": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Bagging": BaggingRegressor(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsRegressor(),
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet Regression": ElasticNet(),
    "Bayesian Ridge Regression": BayesianRidge(),
    "Huber Regressor": HuberRegressor(),
    "SGD Regressor": SGDRegressor(),
    "XGBoost Regressor": XGBRegressor(),
    "LightGBM Regressor": LGBMRegressor(),
    "CatBoost Regressor": CatBoostRegressor(),
    "Neural Network": MLPRegressor(random_state=42),
    "RANSAC Regressor": RANSACRegressor(random_state=42),
    "Gaussian Mixture Model": GaussianMixture(),
    "Kernel Ridge Regression": KernelRidge(),
    "Transformed Target Regression": TransformedTargetRegressor(),
    "Quantile Transformer": make_pipeline(QuantileTransformer(), LinearRegression())
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    accuracy = sum(abs(y_test - predictions) <= 0.05 * y_test) / len(y_test) * 100

    print("\n{}:".format(name))
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("R-squared:", r2)
    print("Accuracy (within 5%): {:.2f}%".format(accuracy))
    
    # Scatter plot of expected vs predicted values
    plt.figure()
    plt.scatter(y_test, predictions)
    plt.title("Expected vs Predicted Values - {}".format(name))
    plt.xlabel("Expected Values")
    plt.ylabel("Predicted Values")
    plt.show()
