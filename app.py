import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load trained model
with open("car_price_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("🚗 Used Car Price Prediction 💰")

# ================== Manual Input Section ==================
st.header("Predict Price Manually")

# User inputs
year = st.number_input("Enter Car Year:", min_value=1990, max_value=2024, value=2015)
mileage = st.number_input("Enter Mileage (miles):", min_value=0, value=50000)
engine_size = st.number_input("Enter Engine Size (L):", min_value=0.5, max_value=8.0, value=2.0)
horsepower = st.number_input("Enter Horsepower:", min_value=50, max_value=1000, value=150)
fuel_type = st.selectbox("Select Fuel Type:", ["Gasoline", "Diesel", "Electric", "Hybrid"])
transmission = st.selectbox("Select Transmission:", ["Automatic", "Manual"])
clean_title = st.radio("Clean Title?", ["Yes", "No"])
accidents = st.number_input("Number of Accidents Reported:", min_value=0, value=0)

# Convert categorical inputs to numbers
fuel_type_map = {"Electric": 0, "Diesel": 1, "Hybrid": 2, "Gasoline": 3}
transmission_map = {"Manual": 0, "Automatic": 1}
clean_title_map = {"Yes": 1, "No": 0}

# Prepare input data
input_data = np.array([[year, mileage, fuel_type_map[fuel_type], engine_size, horsepower, 
                        transmission_map[transmission], accidents, clean_title_map[clean_title]]])

# Predict price
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ${prediction[0]:,.2f}")

# ================== File Upload Section ==================
st.header("Upload Dataset for Predictions")

uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Remove extra spaces in column names
    df.rename(columns=lambda x: x.strip(), inplace=True)

    # Display first few rows
    st.write("### Preview of Uploaded Data:")
    st.write(df.head())

    # Check for missing values
    st.write("### Missing Values in Each Column:")
    st.write(df.isnull().sum())

    # Required columns
    required_cols = {"Year", "Mileage", "Fuel Type", "Engine Size (L)", "Horsepower", "Transmission", "Accidents Reported", "Clean Title"}

    if required_cols.issubset(df.columns):
        # Handle missing values
        df.dropna(inplace=True)  # Drop rows with missing values

        # Convert categorical columns
        df["Fuel Type"] = df["Fuel Type"].map(fuel_type_map).fillna(-1).astype(int)
        df["Transmission"] = df["Transmission"].map(transmission_map).fillna(-1).astype(int)
        df["Clean Title"] = df["Clean Title"].map(clean_title_map).fillna(-1).astype(int)

        # Ensure correct column order
        df = df[["Year", "Mileage", "Fuel Type", "Engine Size (L)", "Horsepower", "Transmission", "Accidents Reported", "Clean Title"]]

        # Convert to NumPy and predict
        X = df.values
        predictions = model.predict(X)
        df["Predicted Price"] = predictions

        # Display predictions
        st.write("### Predicted Prices:")
        st.write(df[["Year", "Mileage", "Fuel Type", "Transmission", "Predicted Price"]])

        # Option to download results
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
    else:
        st.error("Uploaded file does not have the required columns. Please upload a correctly formatted dataset.")
