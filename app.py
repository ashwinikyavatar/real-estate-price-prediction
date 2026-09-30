from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("website_price_model.pkl")


def make_prediction():
    data = request.json

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


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return "Real Estate Price Prediction API is running!"

    return make_prediction()


@app.route("/predict", methods=["POST"])
def predict():
    return make_prediction()


if __name__ == "__main__":
    app.run()