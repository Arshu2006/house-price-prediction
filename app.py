import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("house_price_model.pkl", "rb"))

# Title
st.title("House Price Prediction")

# User inputs
area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
floors = st.number_input("Floors")
yearbuilt = st.number_input("Year Built")

# Predict button
if st.button("Predict Price"):

    # Model expects 12 features (5 inputs + 7 dummy features)
    features = np.array([[area, bedrooms, bathrooms, floors, yearbuilt, 0, 0, 0, 0, 0, 0, 0]])

    prediction = model.predict(features)

    st.success(f"Predicted Price: {prediction[0]}")