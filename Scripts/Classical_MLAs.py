import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
from sklearn.pipeline import Pipeline
class SalesPredictor:

    def __init__(self, train_path, test_df, target_column):
        self.train_path = train_path
        self.test_df = test_df
        self.target_column = target_column
        self.dtype_dict = {
            'Store': 'int64',
            'DayOfWeek': 'int64',
            'Date': 'object',
            'Sales': 'int64',
            'Customers': 'int64',
            'Open': 'int64',
            'Promo': 'int64',
            'StateHoliday': 'str',  # Handle mixed types
            'SchoolHoliday': 'int64'
        }
        self.X_train, self.y_train, self.X_test = self.load_and_merge_data()
        self.X_train_processed, self.X_test_processed, self.feature_names = self.preprocess_data()

    def load_and_merge_data(self):
        train_df = pd.read_csv(self.train_path, dtype=self.dtype_dict, low_memory=False)
        
        print("Loaded train data shape:", train_df.shape)
        print("Train data types:\n", train_df.dtypes)

        # Check for missing values
        print("Missing values in train data:\n", train_df.isnull().sum())

        # Define X and y
        X = train_df.drop(columns=[self.target_column])
        y = train_df[self.target_column]
        
        # Ensure test_df is already a DataFrame
        X_test = self.test_df.drop(columns=['Sales'], errors='ignore')  # Drop Sales if it exists
        print("Loaded test data shape:", X_test.shape)

        return X, y, X_test

    def preprocess_data(self):
        # Ensure both datasets have the same columns
        missing_cols = set(self.X_train.columns) - set(self.X_test.columns)
        
        # If any columns are missing in the test set, add them with default values
        for col in missing_cols:
            self.X_test[col] = 0  # or use an appropriate default value, such as the mean or median
        
        # Identify numeric and categorical columns
        numeric_cols = self.X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = self.X_train.select_dtypes(include=['object']).columns
        
        # Preprocessing for numeric data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])
        
        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        
        # Bundle preprocessing for numeric and categorical data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)])
        
        # Fit the preprocessor on the training data and transform both training and test data
        X_train_processed = preprocessor.fit_transform(self.X_train)
        X_test_processed = preprocessor.transform(self.X_test)
        
        return X_train_processed, X_test_processed, preprocessor.get_feature_names_out()

    def run_random_forest(self):
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_processed, self.y_train)
        rf_predictions = rf_model.predict(self.X_test_processed)
        return rf_predictions, rf_model

    def run_gradient_boosting(self):
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(self.X_train_processed, self.y_train)
        gb_predictions = gb_model.predict(self.X_test_processed)
        return gb_predictions, gb_model

    def run_xgboost(self):
        xg_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        xg_model.fit(self.X_train_processed, self.y_train)
        xg_predictions = xg_model.predict(self.X_test_processed)
        return xg_predictions, xg_model

    def visualize_shap_feature_importance(self, model):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_train_processed)
        shap.summary_plot(shap_values, self.X_train_processed, feature_names=self.feature_names)

    def visualize_predictions(self, y_true, y_pred):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel("Actual Sales")
        plt.ylabel("Predicted Sales")
        plt.title("Actual vs Predicted Sales")
        plt.show()