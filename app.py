import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# Prepare data
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X, y)

st.title("🧬 Diabetes Prediction App")

st.write("Enter patient details:")

preg = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 50.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 0, 120)

if st.button("Predict"):
    input_data = [[preg, glucose, bp, skin, insulin, bmi, dpf, age]]
    result = model.predict(input_data)

    if result[0] == 1:
        st.error("Diabetic")
    else:
        st.success("Not Diabetic")
