import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
# Load the training dataset
data = pd.read_csv("train.csv")

# Display basic information
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])

print("\nFirst 5 rows:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())
print("\nMissing values:")
print(data.isnull().sum().sort_values(ascending=False).head(20))
print("\nData types:")
print(data.dtypes.value_counts())
# Handle missing values

data_clean = data.copy()

# Fill numerical missing values with median
num_cols = data_clean.select_dtypes(include=["int64", "float64"]).columns
data_clean[num_cols] = data_clean[num_cols].fillna(data_clean[num_cols].median())

# Fill categorical missing values with "None"
cat_cols = data_clean.select_dtypes(include=["object"]).columns
data_clean[cat_cols] = data_clean[cat_cols].fillna("None")

print("\nMissing values after cleaning:")
print(data_clean.isnull().sum().sum())
# Separate features and target

X = data_clean.drop("SalePrice", axis=1)
y = data_clean["SalePrice"]

print("\nFeatures shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)
# Convert categorical columns into numerical columns

X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Make sure both datasets have the same columns
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

print("\nAfter encoding:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Predict house prices
y_pred_linear = linear_model.predict(X_test)

# Evaluate the model
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print("\nLinear Regression Results:")
print("MSE:", mse_linear)
print("RMSE:", rmse_linear)
print("R² Score:", r2_linear)
from sklearn.linear_model import Ridge

# Ridge Regression model
ridge_model = Ridge(alpha=10)

# Train the model
ridge_model.fit(X_train, y_train)

# Predict house prices
y_pred_ridge = ridge_model.predict(X_test)

# Evaluate the model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Results:")
print("MSE:", mse_ridge)
print("RMSE:", rmse_ridge)
print("R² Score:", r2_ridge)
from sklearn.linear_model import Lasso

# Lasso Regression model
lasso_model = Lasso(alpha=100)

# Train the model
lasso_model.fit(X_train, y_train)

# Predict house prices
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLasso Regression Results:")
print("MSE:", mse_lasso)
print("RMSE:", rmse_lasso)
print("R² Score:", r2_lasso)
from sklearn.ensemble import RandomForestRegressor

# Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train the model
rf_model.fit(X_train, y_train)

# Predict house prices
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Results:")
print("MSE:", mse_rf)
print("RMSE:", rmse_rf)
print("R² Score:", r2_rf)
from sklearn.ensemble import GradientBoostingRegressor

# Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# Train the model
gb_model.fit(X_train, y_train)

# Predict house prices
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
rmse_gb = np.sqrt(mse_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print("\nGradient Boosting Results:")
print("MSE:", mse_gb)
print("RMSE:", rmse_gb)
print("R² Score:", r2_gb)
# Hyperparameter tuning for Gradient Boosting

param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest Gradient Boosting Parameters:")
print(grid_search.best_params_)

# Best model
best_gb_model = grid_search.best_estimator_

# Prediction
y_pred_best_gb = best_gb_model.predict(X_test)

# Evaluation
mse_best_gb = mean_squared_error(y_test, y_pred_best_gb)
rmse_best_gb = np.sqrt(mse_best_gb)
r2_best_gb = r2_score(y_test, y_pred_best_gb)

print("\nTuned Gradient Boosting Results:")
print("MSE:", mse_best_gb)
print("RMSE:", rmse_best_gb)
print("R² Score:", r2_best_gb)
# Feature Importance

importance = best_gb_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features:")
print(feature_importance.head(15))
# Positive and Negative Price Drivers

correlation = data.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

print("\nPositive Price Drivers:")
print(correlation.head(10))

print("\nNegative Price Drivers:")
print(correlation.tail(10))
# Positive and Negative Price Drivers

correlation = data.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)

print("\nPositive Price Drivers:")
print(correlation.head(10))

print("\nNegative Price Drivers:")
print(correlation.tail(10))
import joblib

joblib.dump(best_gb_model, "house_price_model.pkl")

print("Model saved successfully!")
# 6-feature model for website prediction

website_features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "OverallQual"
]

X_website = data[website_features]
y_website = data["SalePrice"]

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

X_train_web, X_test_web, y_train_web, y_test_web = train_test_split(
    X_website,
    y_website,
    test_size=0.20,
    random_state=42
)

website_model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=2,
    random_state=42
)
# Train the website model
website_model.fit(X_train_web, y_train_web)

# Predict on test data
y_pred_web = website_model.predict(X_test_web)

# Evaluate website model
mse_web = mean_squared_error(y_test_web, y_pred_web)
rmse_web = np.sqrt(mse_web)
r2_web = r2_score(y_test_web, y_pred_web)

print("\nWebsite Model Results:")
print("MSE:", mse_web)
print("RMSE:", rmse_web)
print("R2 Score:", r2_web)
joblib.dump(website_model, "website_model.pkl")
print("Website model saved successfully!")
