import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys 

# Add the path to your EDA_Analysis module
sys.path.append(os.path.abspath(os.path.join('..')))
from EDA_Analysis import EDA_Analysis

#df_train = pd.read_csv('../Data/train.csv', parse_dates=['Date'])
##df_test=pd.read_csv('../Data/test.csv', parse_dates=['Date'])
#f_store=pd.read_csv('../Data/store.csv')
#df_sample_submission=pd.read_csv('../Data/sample_submission.csv')

# Streamlit App
st.title("Exploratory Data Analysis Dashboard")

# File upload for train and test datasets
train_file = st.file_uploader("df_train", type='csv')
test_file = st.file_uploader("df_test", type='csv')

if train_file and test_file:
    # Load datasets
    eda_analysis = EDA_Analysis(train_file, test_file)
    
    # Display datasets
    st.subheader("Training Data")
    st.write(eda_analysis.df_train.head())
    
    st.subheader("Test Data")
    st.write(eda_analysis.df_test.head())
    
    # Compare promo distribution
    if st.button("Compare Promo Distribution"):
        eda_analysis.promo_distribution_comparison()
        st.image('../Images/promo_distribution_comparison.png')

    # Analyze sales around holidays
    if st.button("Analyze Sales Around Holidays"):
        eda_analysis.analyze_sales_around_holidays()
        st.image('../Images/sales_holidays_comparison.png')

    # Analyze holiday sales
    holiday_dates = st.text_input("Enter Holiday Dates (comma-separated, e.g., '2014-12-25, 2015-01-01')")
    if st.button("Analyze Holiday Sales") and holiday_dates:
        holiday_dates = [date.strip() for date in holiday_dates.split(',')]
        eda_analysis.analyze_holiday_sales(holiday_dates)
        st.image('../Images/holiday_sales_comparison.png')

    # Analyze customers sales correlation
    if st.button("Analyze Customers vs Sales Correlation"):
        eda_analysis.analyze_customers_sales_correlation()
        st.image('../Images/customers_sales_correlation.png')

    # Analyze sales trends
    if st.button("Analyze Sales Trends"):
        eda_analysis.analyze_sales_trends()
        st.image('../Images/sales_trends.png')

# Additional configuration for the Streamlit app
st.sidebar.title("Settings")
st.sidebar.write("Configure the analysis options and upload datasets.")