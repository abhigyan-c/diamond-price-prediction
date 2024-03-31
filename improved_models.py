from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the diamonds dataset
diamonds_df = pd.read_csv("diamonds.csv")

# Perform data preprocessing
# Drop unnecessary columns
diamonds_df.drop(columns=['Unnamed: 0'], inplace=True)

# Split data into features and target
X = diamonds_df.drop(columns=['price'])
y = diamonds_df['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numerical and categorical features
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
numeric_transformer = StandardScaler()

categorical_features = ['cut', 'color', 'clarity']
categorical_transformer = OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models to be evaluated
models = [
    ('Random Forest', RandomForestRegressor()),
    ('Bagging', BaggingRegressor(base_estimator=DecisionTreeRegressor())),
    ('Decision Trees', DecisionTreeRegressor()),
    ('XGBoost', XGBRegressor()),
    ('LightGBM', LGBMRegressor()),
    ('CatBoost', CatBoostRegressor(verbose=False))
]

# Perform hyperparameter tuning and evaluate models
results = []
for name, model in models:
    # Create a pipeline with preprocessing and the model
    pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    
    # Define hyperparameters to be tuned
    param_grid = {}
    if name == 'Random Forest':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 10, 20]
        }
    elif name == 'XGBoost':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.3]
        }
    elif name == 'LightGBM':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.3]
        }
    elif name == 'CatBoost':
        param_grid = {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.3]
        }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Evaluate the model
    y_pred = grid_search.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage accuracy within 5%
    within_5_percent = sum(abs(y_test - y_pred) <= 0.05*y_test) / len(y_test) * 100
    
    # Append results
    results.append((name, mae, rmse, r2, within_5_percent))

# Display results
results_df = pd.DataFrame(results, columns=['Model', 'MAE', 'RMSE', 'R-squared', 'Accuracy (within 5%)'])
print(results_df)
