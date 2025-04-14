import streamlit as st
import pickle
import numpy as np

# Load pipeline and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Streamlit Title
st.title("Laptop Price Predictor")

# Brand
company = st.selectbox("Brand", df['Company'].unique())

# Type of Laptop
type = st.selectbox("Type", df["TypeName"].unique())

# RAM
ram = st.selectbox("Ram (in GB)", [2, 4, 6, 8, 16, 24, 32, 64])

# Weight
weight = st.number_input("Weight of the laptop (in kg)", min_value=0.0, format="%.2f")

# Touchscreen
touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

# IPS
ips = st.selectbox("IPS", ["Yes", "No"])

# Screen Size
screen_size = st.number_input("Screen Size (in inches)", min_value=0.0, format="%.2f")

# Resolution
resolution = st.selectbox("Screen Resolution", [
    '1920x1080', '1366x768', '1600x900', '3840x2160',
    '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox("CPU", df["Cpu brand"].unique())

# HDD
hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox("GPU", df['Gpu Brand'].unique())

# OS
os = st.selectbox("Operating System", df['os'].unique())

# Prediction
if st.button("Predict Price"):
    # Convert "Yes" to 1 and "No" to 0
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    # Calculate PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = np.sqrt((X_res ** 2) + (Y_res ** 2)) / screen_size if screen_size > 0 else 0

    # Create input array
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Ensure query is numeric and of the correct shape
    query = np.nan_to_num(query, nan=0.0, posinf=np.finfo(np.float32).max, neginf=np.finfo(np.float32).min)
    query = query.reshape(1, -1)

    # Predict price
    predicted_price = pipe.predict(query)

    # Convert log-transformed price back to actual value
    predicted_price = np.exp(predicted_price[0])

    # Display the result
    st.title(f"The Predicted Price of this configuration is: Rs {int(predicted_price)}")