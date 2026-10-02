from pathlib import Path

import joblib
import pandas as pd

from rest_framework.decorators import api_view
from rest_framework.response import Response

from django.shortcuts import render
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "LinearRegressionModel.pkl"
)

def home(request):
    return render(request, "car/index.html")

@api_view(["POST"])
def predict(request):

    data = request.data

    name = data["name"]
    company = data["company"]
    year = data["year"]
    kms_driven = data["kms_driven"]
    fuel_type = data["fuel_type"]

    input_data = pd.DataFrame([{
        "name": name,
        "company": company,
        "year": year,
        "kms_driven": kms_driven,
        "fuel_type": fuel_type
    }])

    prediction = model.predict(input_data)

    return Response({
        "predicted_price": float(prediction[0])
    })