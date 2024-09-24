Rossman_Sales_Forcast
Overview This project delves into customer purchasing behavior and store sales prediction, aimed at understanding how promotional campaigns, store openings, and seasonal events influence sales. We also build predictive models to forecast daily store sales, offering actionable insights to help retail management plan and strategize.

Exploration of Customer Purchasing Behavior
Objective: Understand and explore customer behavior across various stores to identify key factors that influence purchasing decisions. The goal is to visualize data, clean the dataset, and extract useful insights.

Features Promo Distribution Comparison: Compares the distribution of promotions between the training and test datasets. Sales Analysis Around Holidays: Analyzes average sales trends before, during, and after holidays. Holiday Sales Analysis: Examines average sales behavior around specified holiday dates within a customizable time window. Customers vs. Sales Correlation: Computes and visualizes the correlation between customer count and sales. Sales Trends: Displays average sales trends over time.

Installation
pip install pandas matplotlib seaborn

Class intilazation
from eda_analysis import EDA_Analysis

Initialize the EDA class with paths to the training and test datasets
eda = EDA_Analysis('path_to_train.csv', 'path_to_test.csv')

Task 2: Classical Machine learning
SalesPredictor is a Python class designed to predict sales based on retail data. It provides functionality to preprocess data, train multiple machine learning models (Random Forest, Gradient Boosting, and XGBoost), visualize feature importance using SHAP, and compare actual vs predicted sales.

Features Data Loading & Merging: Loads and merges training and test data, ensuring correct data types and handling missing values. Data Preprocessing: Preprocesses both numerical and categorical data using imputation, scaling, and one-hot encoding. Model Training & Predictions: Trains and generates predictions using Random Forest, Gradient Boosting, and XGBoost models. SHAP Visualizations: Visualizes feature importance using SHAP values for tree-based models. Prediction Visualization: Compares actual vs predicted sales using scatter plots.

Installation
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap

class intilazation
rom sales_predictor import SalesPredictor

Initialize the SalesPredictor with paths and target column
predictor = SalesPredictor(train_path='path_to_train.csv', test_df=test_data, target_column='Sales')

Notes Ensure that the train_path and test_df datasets contain the necessary columns as specified in the dtype_dict. The class handles missing values by imputing the mean for numeric columns and the most frequent value for categorical columns.

License This project is licensed under the MIT License.
