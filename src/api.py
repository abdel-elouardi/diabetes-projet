import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
from src.cleaning import load_data, clean_data, check_duplicates, remove_outliers
from src.model import prepare_data

app = FastAPI()

# Chargement du modèle et du scaler
df = load_data('data/diabetes.csv')
df = clean_data(df)
df = check_duplicates(df)
df = remove_outliers(df)
_, _, _, _, scaler = prepare_data(df)
model = torch.load('models/model.pt', weights_only=False)
model.eval()

class Patient(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def home():
    return {"message": "API Diabète - Bienvenue !"}

@app.post("/predict")
def predict(patient: Patient):
    data = np.array([[
        patient.Pregnancies,
        patient.Glucose,
        patient.BloodPressure,
        patient.SkinThickness,
        patient.Insulin,
        patient.BMI,
        patient.DiabetesPedigreeFunction,
        patient.Age
    ]])
    data = scaler.transform(data)
    tensor = torch.FloatTensor(data)
    with torch.no_grad():
        output = model(tensor).item()
    prediction = 1 if output >= 0.5 else 0
    label = "Diabétique" if prediction == 1 else "Non diabétique"
    return {
        "prediction": prediction,
        "label": label,
        "probabilité": round(output * 100, 2)
    }