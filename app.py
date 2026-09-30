from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load the model made for the website
model = joblib.load("website_price_model.pkl")


@app.route("/")
def home():
    return "Real Estate Price Prediction API is running!"


@app.route("/predict", 
methods=["POST"])
def predict():
    data = request.json

    # Get values from the website
    living_area = float(data["living_area"])
    bedrooms = float(data["bedrooms"])
    bathrooms = float(data["bathrooms"])
    year_built = float(data["year_built"])
    garage = float(data["garage"])
    quality = float(data["quality"])

    # Arrange features in the same order used during training
    features = pd.DataFrame([{
        "GrLivArea": 
                living_area,
        "BedroomAbvGr": 
                bedrooms,
        "FullBath": 
                bathrooms,
        "YearBuilt": 
                year_built,
        "GarageCars": 
                 garage,
        "OverallQual":
                 quality
    }])

    # Predict house price
    prediction = model.predict(features)

    return jsonify({
        "predicted_price": round(float(prediction[0]), 2)
    })


if __name__ == "__main__":
    app.run(debug=True)