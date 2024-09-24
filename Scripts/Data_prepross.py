import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import shap

class SalesPredictionPipeline:
    def __init__(self, train_file, test_file, submission_file):
        # Load the datasets
        self.train_df = pd.read_csv(train_file)
        self.test_df = pd.read_csv(test_file)
        self.submission_df = pd.read_csv(submission_file)

        # Define models
        self.models = {
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'XGBoost': xgb.XGBRegressor(),
            'KNeighbors': KNeighborsRegressor()
        }

    def handle_missing_data(self):
        """Handle missing data in train and test datasets"""
        # Handle missing data in training set
        self.train_df['Customers'].fillna(self.train_df['Customers'].mean(), inplace=True)
        self.train_df['Promo'].fillna(0, inplace=True)  # Assuming missing Promo means no promo
        
        # Handle missing data in test set
        self.test_df['Open'].fillna(1, inplace=True)  # Assuming missing Open means store is open
        
        # Drop rows with too many missing values (if any)
        self.train_df.dropna(subset=['Sales'], inplace=True)
        
        print("Missing data handled successfully.")

    def merge_submission_data(self):
        """Merge test data with submission data to get target Sales column"""
        self.test_df = self.test_df.merge(self.submission_df[['Id', 'Sales']], on='Id', how='left')

    def feature_engineering(self):
        """Create new features such as Year, Month, and Day from Date, and drop the Date column"""
        # Feature engineering for train data
        self.train_df['Year'] = pd.to_datetime(self.train_df['Date']).dt.year
        self.train_df['Month'] = pd.to_datetime(self.train_df['Date']).dt.month
        self.train_df['Day'] = pd.to_datetime(self.train_df['Date']).dt.day

        # Feature engineering for test data
        self.test_df['Year'] = pd.to_datetime(self.test_df['Date']).dt.year
        self.test_df['Month'] = pd.to_datetime(self.test_df['Date']).dt.month
        self.test_df['Day'] = pd.to_datetime(self.test_df['Date']).dt.day

        # Drop the Date column
        self.train_df.drop(columns=['Date'], inplace=True)
        self.test_df.drop(columns=['Date'], inplace=True)

        print("Feature engineering completed.")

    def preprocess(self):
        """Perform preprocessing including feature scaling and encoding"""
        # Separate features and target variable from train data
        X = self.train_df.drop(columns=['Sales', 'Customers'])  # Dropping Customers as it's highly correlated with Sales
        y = self.train_df['Sales']

        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessing pipeline for numerical and categorical features
        numeric_features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Year', 'Month', 'Day']
        categorical_features = ['StateHoliday']

        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_features),
                          ('cat', categorical_transformer, categorical_features)])

        # Fit and transform the training data
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_val_scaled = preprocessor.transform(X_val)

        print("Preprocessing completed.")

        return X_train_scaled, X_val_scaled, y_train, y_val, preprocessor

    def train_without_tuning(self, X_train, y_train, X_val, y_val):
        """Train models without hyperparameter tuning and evaluate on validation data with RMSE, MSE, and MAE"""
        metrics = {}

        for model_name, model in self.models.items():
            print(f"Training {model_name} without tuning...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)

            metrics[model_name] = {'RMSE': rmse, 'MSE': mse, 'MAE': mae}
            print(f"{model_name} - RMSE: {rmse}, MSE: {mse}, MAE: {mae}")

        return metrics

    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning using RandomizedSearchCV for all models"""
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.01],
                'max_depth': [3, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.1, 0.01],
                'max_depth': [3, 5, 10]
            },
            'KNeighbors': {
                'n_neighbors': [3, 5, 10],
                'weights': ['uniform', 'distance']
            }
        }

        best_models = {}

        for model_name, model in self.models.items():
            print(f"Tuning {model_name}...")
            random_search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[model_name],
                                               n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
            random_search.fit(X_train, y_train)
            best_models[model_name] = random_search.best_estimator_
            print(f"Best parameters for {model_name}: {random_search.best_params_}")

        return best_models

    def evaluate_model(self, X_val, y_val, models):
        """Evaluate the performance of models on validation data using RMSE, MSE, and MAE"""
        metrics = {}
        for name, model in models.items():
            y_pred = model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mse = mean_squared_error(y_val, y_pred)
            mae = mean_absolute_error(y_val, y_pred)

            metrics[name] = {'RMSE': rmse, 'MSE': mse, 'MAE': mae}
            print(f"{name} - RMSE: {rmse}, MSE: {mse}, MAE: {mae}")

        return metrics

    def model_interpretation(self, model, X_val_scaled):
        """Perform model interpretation using SHAP values"""
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val_scaled)

        print("Plotting SHAP summary plot...")
        shap.summary_plot(shap_values, X_val_scaled)