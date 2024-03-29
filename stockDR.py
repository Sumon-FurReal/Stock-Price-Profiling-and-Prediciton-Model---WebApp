
# Imports
# from operator import index
import streamlit as st
from PIL import Image
import plotly.express as px
import pandas as pd
import os
# import joblib
import pickle
import pandas_profiling
import streamlit_pandas_profiling
# import sklearn

# Profiling Imports
from streamlit_pandas_profiling import st_profile_report



# OS Path
if os.path.exists("source.csv"):
    df = pd.read_csv("source.csv", index_col=None)
    ph_df = pd.DataFrame(df)

# Image Source
image = Image.open("pic2.png")


# Flow
with st.sidebar:
    st.image(image)
    st.title("Stock Price Profiling and Prediciton Model  :dart:")
    choice = st.radio(
        "OPTIONS", ["Upload Stock Data", "EDA - Profiling", "Predict Stock Price"])
    st.info("This applicaiton allows you to perform exploratory data analysis(EDA) and predict stock price **NOTE - Please use CSV file**")


if choice == "Upload Stock Data":
    st.title("Upload stock data for modeling   :bar_chart:")
    st.info("Allows users to upload their stock data in CSV format and displays the data in a DataFrame.")
    file = st.file_uploader("Upload Your Dataset Here ")
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv("source.csv", index=None)
        st.dataframe(df)

if choice == "EDA - Profiling":
    st.title("Exploratory Data Analysis - Profiling  :mag:")
    st.info("Performs Exploratory Data Analysis (EDA) using Pandas Profiling and shows an interactive report with insights into the data.")
    if st.button("Begin EDA"):
        profile_df = df.profile_report()
        st_profile_report(profile_df)

if choice == "Predict Stock Price":
    st.title("Stock Predictor :crystal_ball:")
    st.info("Utilizes a pre-trained machine learning model to predict the stock's closing price based on user-provided input features such as Open Price, High Price, Low Price, Close Price, Adj Close Price, and Volume.")
    if os.path.exists("source.csv"):
        loaded_model = pickle.load(open('price_history_model.pkl', 'r'))

        def predict(input_data):
            prediction = loaded_model.predict(input_data)
            return prediction

        st.header("Your Stock Prediction")
        open_price = st.number_input("Open Price", value=0.0)
        high = st.number_input("High Price", value=0.0)
        low = st.number_input("Low Price", value=0.0)
        close = st.number_input("Close Price", value=0.0)
        adj_close = st.number_input("Adj Close Price", value=0.0)
        volume = st.number_input("Volume", value=0)
        if st.button("Predict"):
            input_data = pd.DataFrame({'Open': [open_price],
                                       'High': [high],
                                       'Low': [low],
                                       'Close': [close],
                                       'Adj Close': [adj_close],
                                       'Volume': [volume]})
            if input_data.isnull().values.any():
                st.write('Invalid input data, please enter valid data')
            else:
                prediction = predict(input_data)
                st.write("Predicted Close Price", prediction)
