import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Load training data
data = pd.read_csv("train.csv")


# Features used by our website
features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "YearBuilt",
    "GarageCars",
    "OverallQual"
]

X = data[features]
y = data["SalePrice"]


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Models
models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Lasso Regression": Lasso(alpha=0.01, max_iter=10000),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=2,
        random_state=42
    )
}


# Store results
results = {}


# Train and evaluate each model
for name, algorithm in models.items():

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", algorithm)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results[name] = {
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# Print results
print("\n==============================")
print("MODEL COMPARISON RESULTS")
print("==============================")

for name, result in results.items():

    print("\n" + name)
    print("MSE:", round(result["MSE"], 2))
    print("RMSE:", round(result["RMSE"], 2))
    print("R2 Score:", round(result["R2"], 4))


# Train final model for the website
final_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=2,
        random_state=42
    ))
])


final_model.fit(X_train, y_train)


# Save model
joblib.dump(final_model, "website_price_model.pkl")

print("\n==============================")
print("Website model saved successfully!")
print("==============================")