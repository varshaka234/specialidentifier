import pickle
import streamlit as st
import pandas as pd
from os import path
# from sklearn.datasets import load_iris

st.title('🌸 Iris Flower Species Predictor')

# Inputs
petal_length = st.number_input("Petal length (cm)", min_value=1.0, max_value=6.9)
petal_width  = st.number_input("Petal width (cm)", min_value=0.1, max_value=2.5)
sepal_length = st.number_input("Sepal length (cm)", min_value=4.3, max_value=7.9)
sepal_width  = st.number_input("Sepal width (cm)", min_value=2.0, max_value=4.4)

df_user_input = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=['sepal_length','sepal_width','petal_length','petal_width']
)

st.write("Your input:", df_user_input)

# Load model
model_path = path.join("Model", "iris_classifier.pkl")
with open(model_path, "rb") as file:
    iris_model = pickle.load(file)

# Predict
dict_species = {0:'setosa', 1:'versicolor', 2:'virginica'}

if st.button("Predict species"):
    if (petal_length is None) or (petal_width is None) or (sepal_length is None) or (sepal_width is None):
        st.write("⚠️ Please fill all values")
    else:
        predicted_species = iris_model.predict(df_user_input)
        st.success(f"🌼 The species is **{dict_species[predicted_species[0]]}**")
