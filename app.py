from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)

# Allow GitHub Pages to access this API
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"]
)

model = joblib.load("website_price_model.pkl")


def make_prediction():
    data = request.get_json()

    living_area = float(data["living_area"])
    bedrooms = float(data["bedrooms"])
    bathrooms = float(data["bathrooms"])
    year_built = float(data["year_built"])
    garage = float(data["garage"])
    quality = float(data["quality"])

    features = pd.DataFrame([{
        "GrLivArea": living_area,
        "BedroomAbvGr": bedrooms,
        "FullBath": bathrooms,
        "YearBuilt": year_built,
        "GarageCars": garage,
        "OverallQual": quality
    }])

    prediction = model.predict(features)

    return jsonify({
        "predicted_price": round(float(prediction[0]), 2)
    })


@app.route("/", methods=["GET", "POST", "OPTIONS"])
def home():
    if request.method == "GET":
        return "Real Estate Price Prediction API is running!"

    if request.method == "OPTIONS":
        return "", 204

    return make_prediction()


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    return make_prediction()


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


if __name__ == "__main__":
    app.run()