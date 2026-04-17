import streamlit as st
import torch
import numpy as np
from src.cleaning import load_data, clean_data, check_duplicates, remove_outliers
from src.model import prepare_data

# Chargement du modèle et du scaler
df = load_data('data/diabetes.csv')
df = clean_data(df)
df = check_duplicates(df)
df = remove_outliers(df)
_, _, _, _, scaler = prepare_data(df)
model = torch.load('models/model.pt', weights_only=False)
model.eval()

# Interface
st.title("🩺 Prédiction du Diabète")
st.write("Entrez les informations du patient :")

pregnancies = st.number_input("Grossesses", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Pression artérielle", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Épaisseur de peau", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insuline", min_value=0, max_value=900, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input("Fonction pedigree diabète", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Âge", min_value=1, max_value=120, value=30)

if st.button("🔍 Prédire"):
    data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                      insulin, bmi, dpf, age]])
    data = scaler.transform(data)
    tensor = torch.FloatTensor(data)
    with torch.no_grad():
        output = model(tensor).item()
    prediction = 1 if output >= 0.5 else 0
    probabilite = round(output * 100, 2)

    if prediction == 1:
        st.error(f"⚠️ Diabétique — Probabilité : {probabilite}%")
    else:
        st.success(f"✅ Non diabétique — Probabilité : {probabilite}%")